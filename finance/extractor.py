"""发票字段抽取 —— 允许出错的那一层
=====================================

本模块是整个项目里**唯一允许 LLM 参与**的地方，也是唯一**允许出错**的地方。
理由很简单：把非结构化的发票变成结构化字段，本来就不是确定性任务。
但出错不要紧 —— 下一层的规则引擎会逐条校验，错了会被抓住。
这正是分层的意义：**把不确定性圈在一层里**。

两级抽取策略
------------
1. **PDF 文本层（确定性，不联网）** —— 电子发票 PDF 自带文本层，``pypdf``
   直接读出来用正则解析。可复现、零成本、零延迟。**主演示路径走这条。**
2. **视觉模型（兜底，联网）** —— 拍照件、扫描件、或者版式对不上的，
   交给 qwen-vl-max 看图转 JSON。默认**关闭**（``use_vision=False``），
   因为演示时不该依赖网络。扫描件的 PDF 会先把页面内嵌的位图抠出来
   （见 :func:`extract_embedded_images`），**绝不把整个 PDF 当图片发出去**。

三个必须绕开的坑（都是实测出来的）
----------------------------------
- ``tools.vision`` 的默认 ``max_tokens`` 是 1024，十几个字段的中文 JSON
  会被截断。本模块显式传更大的值，并且做"截断后二次解析"容错。
- ``tools.vision.mime_of()`` 不认识的扩展名一律回落成 ``image/png``。把 ``.pdf``
  直接递过去，等于把 PDF 文件流贴上 PNG 的标签发给模型 —— 那条路是死的
  （不报错，只是永远识别不出来）。所以 PDF 必须先抠出真正的位图。
- 历史上这个模块用 ``raise SystemExit`` 报错（继承 ``BaseException``，
  ``except Exception`` 兜不住，缺个密钥就能把 uvicorn 进程带走）。
  现已改为 ``VisionError``；本层仍然做兜底转换，因为这里是"允许出错的那一层"，
  谁抛的异常都不该变成一次 500。
"""

from __future__ import annotations

import json
import re
import struct
import tempfile
import zlib
from datetime import date
from pathlib import Path
from typing import Any

from .models import Invoice, parse_money

VISION_PROMPT = """请识别这张发票，只返回一个 JSON 对象，不要任何解释文字、不要 markdown 代码块。
JSON 的键固定为：
{
  "invoice_type": "发票类型（票面标题），如 电子发票（普通发票）",
  "invoice_code": "发票代码，没有则空字符串",
  "invoice_number": "发票号码",
  "issue_date": "开票日期，格式 YYYY-MM-DD",
  "buyer_name": "购买方名称",
  "buyer_tax_id": "购买方纳税人识别号",
  "seller_name": "销售方名称",
  "seller_tax_id": "销售方纳税人识别号",
  "item_name": "项目名称，原样照抄含 * 号",
  "amount": "金额（不含税），纯数字字符串",
  "tax_rate": "税率，如 3%",
  "tax_amount": "税额，纯数字字符串",
  "total": "价税合计（小写），纯数字字符串",
  "total_in_words": "价税合计（大写）",
  "remark": "备注栏内容，没有则空字符串"
}
看不清或票面没有的字段填空字符串，不要猜。"""


class ExtractionError(Exception):
    """抽取失败。调用方应降级为 WARN + 人工补录，而不是让整单崩掉。"""


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def extract(
    path: str | Path,
    *,
    use_vision: bool = False,
    vision_timeout: int = 20,
) -> Invoice:
    """从发票文件抽取字段。

    :param use_vision: PDF 文本层解析失败时，是否允许调用视觉模型兜底
    :raises ExtractionError: 两级都失败
    """
    p = Path(path)
    if not p.is_file():
        raise ExtractionError(f"文件不存在: {p}")

    suffix = p.suffix.lower()

    if suffix == ".pdf":
        try:
            return extract_from_pdf(p)
        except ExtractionError as pdf_exc:
            if not use_vision:
                raise
            # PDF 文本层读不出来（扫描件？）—— 抠出内嵌位图再转视觉兜底
            return _extract_pdf_via_vision(p, pdf_exc, vision_timeout)

    # 电子发票的结构化形态：优先直接解析（零 OCR、零模型、零猜）
    if suffix == ".xml":
        return extract_from_xml(p)
    if suffix == ".ofd":
        return extract_from_ofd(p)

    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
        return extract_from_image(p, timeout=vision_timeout)

    raise ExtractionError(
        f"不支持的文件类型: {suffix}（支持 xml / ofd / pdf 与常见图片格式）"
    )


# --------------------------------------------------------------------------
# 一级：PDF 文本层（确定性）
# --------------------------------------------------------------------------


def extract_from_pdf(path: str | Path, source_file: str = "") -> Invoice:
    """用 pypdf 读文本层，再用正则解析字段。**不联网、可复现。**"""
    p = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("未安装 pypdf，无法解析 PDF") from exc

    try:
        reader = PdfReader(str(p))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"PDF 读取失败: {type(exc).__name__}: {exc}") from exc

    text = _normalize_text(text)
    if not text.strip():
        raise ExtractionError("PDF 没有可提取的文本层（可能是整页只有一张图的扫描件）")

    invoice = _parse_invoice_text(text)
    invoice.raw_text = text
    invoice.source_file = source_file or p.name
    invoice.extraction_method = "pdf_text"
    return invoice


# --------------------------------------------------------------------------
# 二级：视觉模型（兜底）
# --------------------------------------------------------------------------


def extract_from_image(path: str | Path, timeout: int = 20) -> Invoice:
    """调 qwen-vl-max 看图，要求返回 JSON。

    **注意** 这里仍然兜住 ``BaseException``：``tools.vision`` 现在抛的是普通
    ``VisionError``，但抽取层是"允许出错的那一层"，任何来自第三方的异常
    （含 ``SystemExit`` 这种会带走 uvicorn 的）都不该越过这一层。
    """
    p = Path(path)
    try:
        from tools.vision import describe
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("无法导入 tools.vision 模块") from exc

    try:
        raw = describe(str(p), prompt=VISION_PROMPT, timeout=timeout, max_tokens=2048)
    except BaseException as exc:  # noqa: BLE001 —— 刻意兜住一切，包括 SystemExit
        raise ExtractionError(f"视觉模型调用失败: {type(exc).__name__}: {exc}") from exc

    if not isinstance(raw, str) or not raw.strip():
        raise ExtractionError("视觉模型没有返回内容")

    data = _loads_lenient(raw)
    if data is None:
        raise ExtractionError(f"视觉模型返回的不是合法 JSON，无法解析: {raw[:200]}")

    invoice = _invoice_from_mapping(data)
    invoice.raw_text = raw
    invoice.source_file = p.name
    invoice.extraction_method = "vision"
    return invoice


def _extract_pdf_via_vision(
    pdf_path: Path, pdf_exc: ExtractionError, timeout: int
) -> Invoice:
    """PDF 走视觉兜底：先把页面内嵌的位图抠出来，再逐张送去识别。

    抠不出图就**明确报错**，不再像过去那样把整份 PDF 当 PNG 发出去。
    """
    images = extract_embedded_images(pdf_path)
    if not images:
        raise ExtractionError(
            f"{pdf_exc}；并且在这个 PDF 里没有找到内嵌位图，视觉兜底无从下手。"
            "请改传扫描件的图片文件（jpg/png）。"
        ) from pdf_exc

    last_exc: ExtractionError | None = None
    for index, (data, suffix) in enumerate(images, start=1):
        tmp_path: Path | None = None
        try:
            # describe_image 只认文件路径，所以先落一个带正确扩展名的临时文件 ——
            # 扩展名不是小事，mime_of() 就是靠它决定发给模型的是 image/png 还是 image/jpeg。
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
                fh.write(data)
                tmp_path = Path(fh.name)
            return extract_from_image(tmp_path, timeout=timeout)
        except ExtractionError as exc:
            last_exc = exc
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    raise ExtractionError(
        f"PDF 内嵌的 {len(images)} 张位图都没能识别成功：{last_exc}"
    ) from last_exc


# --------------------------------------------------------------------------
# PDF 内嵌位图 —— 扫描件兜底的必经之路
# --------------------------------------------------------------------------

# pypdf 的 ``page.images`` 需要 Pillow，本项目不引入新依赖，所以直接读 XObject。
_JPEG_FILTER = "/DCTDecode"
_SKIP_FILTERS = ("/JBIG2Decode",)
_MAX_XOBJECT_DEPTH = 4


def extract_embedded_images(path: str | Path) -> list[tuple[bytes, str]]:
    """抽出 PDF 页面里的内嵌位图，返回 ``[(字节, 扩展名)]``，按页序排列。

    扫描件 PDF 的形态就是"每页一张图、文字层为空"，这时唯一的出路是把那张图
    抠出来交给视觉模型。**不能把整个 PDF 当图片发出去**：``mime_of()`` 对
    ``.pdf`` 会回落成 ``image/png``，等于把 PDF 文件流贴上 PNG 的标签发给模型，
    不报错，只是永远识别不出来。

    两种处理方式：

    - ``DCTDecode``（JPEG，扫描件最常见）：字节原样透传 —— pypdf 对图像滤镜
      不解码，取出来的就是 JPEG 本身。
    - 其余位图：pypdf 能解码成原始样本的（Flate / LZW / RunLength / CCITT），
      若是 8 位灰度或 RGB，就用 zlib 自己封成 PNG。CMYK、索引色、1/2/4 位
      这些不转换 —— 宁可少一条路，也不发一张颜色错乱的图出去。
    """
    from pypdf import PdfReader

    p = Path(path)
    try:
        reader = PdfReader(str(p))
    except Exception as exc:  # noqa: BLE001 —— pypdf 的解析异常类型很杂
        raise ExtractionError(f"PDF 无法解析：{type(exc).__name__}: {exc}") from exc

    found: list[tuple[bytes, str]] = []
    for page in reader.pages:
        _collect_page_images(page.get("/Resources"), found, depth=0)
    return found


def _collect_page_images(resources: Any, out: list[tuple[bytes, str]], depth: int) -> None:
    """遍历资源字典里的 XObject；表单 XObject 会再往里钻一层。"""
    if resources is None or depth > _MAX_XOBJECT_DEPTH:
        return
    xobjects = resources.get("/XObject")
    if xobjects is None:
        return
    for obj in xobjects.values():
        obj = obj.get_object()
        subtype = obj.get("/Subtype")
        if subtype == "/Image":
            converted = _image_bytes(obj)
            if converted is not None:
                out.append(converted)
        elif subtype == "/Form":
            _collect_page_images(obj.get("/Resources"), out, depth + 1)


def _image_bytes(obj: Any) -> tuple[bytes, str] | None:
    """把一个图像 XObject 转成 ``(字节, 扩展名)``；做不到就返回 None。"""
    filters = obj.get("/Filter") or []
    if isinstance(filters, str):
        filters = [filters]
    filters = [str(f) for f in filters]

    if filters and filters[-1] == _JPEG_FILTER:
        # JPEG 原样透传：pypdf 的 DCTDecode 是恒等解码，取出来就是 JPEG 字节流。
        return bytes(obj.get_data()), ".jpg"

    if any(f in _SKIP_FILTERS for f in filters):
        return None

    width = obj.get("/Width")
    height = obj.get("/Height")
    bits = obj.get("/BitsPerComponent", 8)
    color_space = obj.get("/ColorSpace")
    if not width or not height or int(bits) != 8:
        return None
    channels = {"/DeviceGray": 1, "/DeviceRGB": 3}.get(str(color_space))
    if channels is None:
        return None

    try:
        samples = bytes(obj.get_data())
    except Exception:  # noqa: BLE001 —— 解码失败只意味着这张图用不了
        return None
    if len(samples) < int(width) * int(height) * channels:
        return None
    return _png_from_samples(samples, int(width), int(height), channels), ".png"


def _png_from_samples(samples: bytes, width: int, height: int, channels: int) -> bytes:
    """把裸像素样本封成最小 PNG（每行 filter type 0）。纯标准库，无第三方依赖。"""
    stride = width * channels
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw += samples[y * stride : (y + 1) * stride]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2 if channels == 3 else 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


def _loads_lenient(raw: str) -> dict[str, Any] | None:
    """容忍 markdown 围栏与前后废话的 JSON 解析。

    三级降级：直接 loads -> 剥 ``` 围栏 -> 截取第一个 { 到最后一个 }。
    视觉模型返回 JSON 时装在代码块里是常态，不是异常。
    """
    candidates = [raw.strip()]

    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
        candidates.append(stripped.strip())

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _invoice_from_mapping(d: dict[str, Any]) -> Invoice:
    """把视觉模型返回的 dict 转成 Invoice，逐个字段容错。"""
    def s(key: str) -> str:
        v = d.get(key)
        return "" if v is None else str(v).strip()

    def m(key: str):
        v = d.get(key)
        if v is None or str(v).strip() == "":
            return None
        try:
            return parse_money(v)
        except ValueError:
            return None

    return Invoice(
        invoice_code=s("invoice_code"),
        invoice_number=s("invoice_number"),
        invoice_type=s("invoice_type"),
        issue_date=_parse_date_flexible(s("issue_date")),
        buyer_name=s("buyer_name"),
        buyer_tax_id=s("buyer_tax_id"),
        seller_name=s("seller_name"),
        seller_tax_id=s("seller_tax_id"),
        item_name=s("item_name"),
        amount=m("amount"),
        tax_rate=s("tax_rate"),
        tax_amount=m("tax_amount"),
        total=m("total"),
        total_in_words=s("total_in_words"),
        remark=s("remark"),
    )


# --------------------------------------------------------------------------
# 结构化直取：电子发票 XML / OFD（零 OCR、零模型）
# --------------------------------------------------------------------------


def extract_from_xml(path: str | Path, source_file: str = "") -> Invoice:
    """电子发票 XML（数电票 / 开票平台导出）直接解析。

    轻量实现，边界如实说：兼容常见**中英文标签**（标签对不上就抽稀，
    抽不到的字段由规则引擎判"信息缺失转人工"）；标签语义随开票平台有差异
    （TotalAmount 有的家是价税合计、有的是不含税），**对不上的会被 R013
    的三要素勾稽抓出来标冲突**，由人工按原件更正 —— 不猜。
    """
    p = Path(path)
    return _parse_invoice_xml(
        p.read_text(encoding="utf-8", errors="ignore"), source_file or p.name, "xml"
    )


def extract_from_ofd(path: str | Path, source_file: str = "") -> Invoice:
    """OFD（增值税电子发票）= ZIP 容器，取内嵌结构化 XML 走同一条解析路。

    边界：只读**内嵌结构化数据**（Invoice/Bill 类 XML）；纯版式 OFD
    （文字画在 ContentStream 里）本实现不渲染 —— 抽不到就报错转
    手动/视觉兜底，**不猜**。
    """
    import zipfile

    p = Path(path)
    try:
        with zipfile.ZipFile(p) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
            # 优先取名字像发票数据的，其次取最大的 XML
            names.sort(
                key=lambda n: (
                    ("invoice" not in n.lower() and "bill" not in n.lower()),
                    -zf.getinfo(n).file_size,
                )
            )
            last_exc: ExtractionError | None = None
            for n in names:
                try:
                    return _parse_invoice_xml(
                        zf.read(n).decode("utf-8", errors="ignore"),
                        source_file or p.name,
                        "ofd",
                    )
                except ExtractionError as exc:
                    last_exc = exc
                    continue
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"OFD 无法打开（不是有效 ZIP 容器）：{exc}") from exc
    raise ExtractionError(
        f"OFD 里没有可用的结构化发票数据（{last_exc or '未找到 XML'}）。"
        "请改传 PDF/图片，或手动填写。"
    )


def _parse_invoice_xml(text: str, source_file: str, method: str) -> Invoice:
    """XML 文本 -> Invoice。抽到关键字段（号码/金额）之一才算成功。"""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ExtractionError(f"XML 无法解析：{exc}") from exc

    values: dict[str, str] = {}
    for node in root.iter():
        tag = node.tag.split("}")[-1]          # 命名空间随便剥
        if node.text and node.text.strip():
            values.setdefault(tag, node.text.strip())
            values.setdefault(tag.replace(" ", "").lower(), node.text.strip())

    def pick(*names: str) -> str:
        for n in names:
            if n in values:
                return values[n]
            if n.replace(" ", "").lower() in values:
                return values[n.replace(" ", "").lower()]
        return ""

    invoice = Invoice(
        invoice_code=pick("InvoiceCode", "发票代码"),
        invoice_number=pick("InvoiceNumber", "发票号码", "BillNumber", "票据号码", "Number"),
        invoice_type=pick("InvoiceType", "发票类型", "BillType", "票据类型"),
        issue_date=_parse_date_flexible(pick("IssueDate", "开票日期", "Date")),
        buyer_name=pick("BuyerName", "购买方名称", "Buyer", "抬头"),
        buyer_tax_id=pick("BuyerTaxId", "BuyerTaxNo", "购买方纳税人识别号", "购买方税号"),
        seller_name=pick("SellerName", "销售方名称", "Seller"),
        seller_tax_id=pick("SellerTaxId", "SellerTaxNo", "销售方纳税人识别号", "销售方税号"),
        item_name=pick("ItemName", "项目名称", "GoodsName", "商品或服务名称"),
        amount=_safe_money(pick("Amount", "AmountWithoutTax", "不含税金额", "金额")),
        tax_rate=pick("TaxRate", "税率"),
        tax_amount=_safe_money(pick("TaxAmount", "税额")),
        total=_safe_money(pick("TotalAmount", "AmountWithTax", "价税合计", "合计金额", "合计")),
        total_in_words=pick("TotalInWords", "价税合计大写", "大写金额", "大写"),
        remark=pick("Remark", "备注"),
        raw_text=text[:2000],
        source_file=source_file,
        extraction_method=method,
    )
    if not invoice.invoice_number and invoice.total is None:
        raise ExtractionError("XML 里找不到发票关键字段（号码/金额），版式不受支持")
    return invoice


# --------------------------------------------------------------------------
# 置信度标注（AI 预填 + 人工确认的中间层）
# --------------------------------------------------------------------------

#: 关键字段：必须与原件/查验结果一致 —— 修改须留"已核对原件"声明。
#: **发票类型也在内**：它是规则适用范围（票种路由）的开关，改它等于改判定口径
#: （B1 教训：S04 一张抬头违规的票，改 invoice_type 为"火车票"曾直接翻案成 APPROVED）。
CRITICAL_FIELDS = ("total", "buyer_tax_id", "invoice_number", "invoice_type")


def assess_confidence(invoice: Invoice) -> dict[str, str]:
    """给每个票面字段标注置信度，前端按色标呈现（绿/黄/红/灰）。

    规则全部**确定性（零模型）**：

    - ``high``（绿）：结构化来源（PDF 文本层 / XML / OFD）抽到的完整字段
      —— 可一键确认；
    - ``low``（黄）：**视觉/OCR 抽取的一律进这档**（OCR 有误差，必须人工核对）——
      这补上了「视觉抽取路径没有人工核对卡点」的缺口；
    - ``conflict``（红）：字段之间**对不上**（大小写金额不一致、
      不含税+税额≠价税合计）—— 票面自相矛盾或被篡改的典型特征，
      **必须人工按原件更正**；
    - ``missing``（灰）：没抽到，人工补。

    查验 API 接入后（interfaces.InvoiceVerifier），关键三项可再上调 —— Phase 2。
    """
    from .models import parse_chinese_amount, parse_money

    vision = (invoice.extraction_method or "") in {"vision"}

    def base(ok: bool) -> str:
        if not ok:
            return "missing"
        return "low" if vision else "high"

    conf = {
        "invoice_code": base(bool(invoice.invoice_code)),
        "invoice_number": base(bool(invoice.invoice_number)),
        "invoice_type": base(bool(invoice.invoice_type)),
        "issue_date": base(invoice.issue_date is not None),
        "buyer_name": base(bool(invoice.buyer_name)),
        "buyer_tax_id": base(bool(invoice.buyer_tax_id)),
        "seller_name": base(bool(invoice.seller_name)),
        "seller_tax_id": base(bool(invoice.seller_tax_id)),
        "item_name": base(bool(invoice.item_name)),
        "amount": base(invoice.amount is not None),
        "tax_rate": base(bool(invoice.tax_rate)),
        "tax_amount": base(invoice.tax_amount is not None),
        "total": base(invoice.total is not None),
        "total_in_words": base(bool(invoice.total_in_words)),
        "remark": base(bool(invoice.remark)),
    }

    # ---- 冲突检测：比"字段有没有"更重要 —— 这是防篡改的眼睛 ----
    words = (
        parse_chinese_amount(invoice.total_in_words) if invoice.total_in_words else None
    )
    if words is not None and invoice.total is not None and words != invoice.total:
        conf["total"] = conf["total_in_words"] = "conflict"
    if None not in (invoice.amount, invoice.tax_amount, invoice.total):
        if (
            parse_money(invoice.amount) + parse_money(invoice.tax_amount)
            != parse_money(invoice.total)
        ):
            conf["amount"] = conf["tax_amount"] = conf["total"] = "conflict"
    return conf


# --------------------------------------------------------------------------
# 文本解析（从 PDF 文本层到 Invoice）
# --------------------------------------------------------------------------

_RE_NUMBER = re.compile(r"发票号码[：:\s]*([0-9]{8,})")
_RE_DATE = re.compile(r"开票日期[：:\s]*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_RE_DATE_ALT = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_RE_BUYER = re.compile(r"购买方名称[：:\s]*([^\n]+)")
_RE_SELLER = re.compile(r"销售方名称[：:\s]*([^\n]+)")
_RE_TAXID = re.compile(r"纳税人识别号[）)]?[：:\s]*([0-9A-Za-z]{15,20})")
_RE_TOTAL_SMALL = re.compile(r"[（(]小写[）)][￥¥]?\s*([0-9,]+\.\d{2})")
_RE_TOTAL_WORDS = re.compile(r"价税合计[（(]大写[）)][：:\s]*([^\n（(]+)")
_RE_TOTAL_WORDS_ALT = re.compile(r"[（(]大写[）)][：:\s]*([^\n（(]+)")
_RE_REMARK = re.compile(r"备注[：:\s]*([^\n]*)")
_RE_ITEM_LINE = re.compile(
    r"(\*[^*\n]+\*[^\s\n]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+%)\s+([\d.]+)"
)
_RE_TYPE_LINE = re.compile(r"^\s*(电子发票[（(][^）)]*[）)]|增值税[^\n]{0,12}发票|数电票)\s*$", re.MULTILINE)


# 票面排版给标题加字间距（letter-spacing）时，PDF 抽取层会把字与字之间的空隙
# 当成一个空格，于是一行「电子发票（普通发票）」被抽成
# 「电 子 发 票 （ 普 通 发 票 ）」，任何按整行匹配的正则都会失配。
# 中文正文不用空格分词，非 ASCII 字符之间的空白一定是排版产物，去掉才能做字段匹配。
#
# 三个坑，都是实测踩出来的：
#   1. 用「非空白取反」表达「空白但不含换行」，而不是列举空格字符。列举法会漏掉
#      全角空格等 Unicode 空白；图省事写成通配的空白类又会把换行一起吃掉，
#      相邻两行粘成一行，按行锚定的正则（如 _RE_TYPE_LINE）会全部失效。
#   2. 必须在「全角转半角」之前做。半角括号落在 ASCII 区间，顺序反了就会残留
#      「票 ( 普」这种半角括号两侧的空格，等于白做。
#   3. 只删非 ASCII 字符之间的空白，ASCII 之间的空格原样保留 ——
#      项目行「*住宿服务*住宿费 3 550.00」靠这些空格分词，不能动。
#
# 历史事故：样本票标题带字间距，新版 pypdf 抽出来是空格分隔的，发票类型抽成空串，
# R009 对空值判 FAIL，把一张完全合规的样本票打成 REJECTED。
_RE_LETTER_SPACED = re.compile(r"(?<=[^\x00-\x7f])[^\S\n]+(?=[^\x00-\x7f])")


def _squeeze_letter_spacing(text: str) -> str:
    """去掉字间距造成的空白。设计说明见 _RE_LETTER_SPACED。"""
    return _RE_LETTER_SPACED.sub("", text)


def _normalize_text(text: str) -> str:
    """统一全角/半角符号与空白，让正则不必到处写兼容分支。

    第一步先去掉字间距造成的空白（见 _RE_LETTER_SPACED）：
    它必须发生在全角转半角之前，否则半角括号两侧的空格会留下来。
    """
    text = _squeeze_letter_spacing(text)
    out = text.replace("　", " ")
    out = out.replace("（", "(").replace("）", ")")
    out = out.replace("：", ":")
    # 英文括号还原成正则在用的半角，注意上面的 replace 已处理中文括号
    out = re.sub(r"[ \t]+", " ", out)
    return out


def _parse_invoice_text(text: str) -> Invoice:
    """从发票文本里逐字段抓取。抓不到的字段留空，由规则引擎判定为缺失。"""
    invoice = Invoice()

    m = _RE_TYPE_LINE.search(text)
    if m:
        # 还原成票面原本的中文括号写法
        invoice.invoice_type = m.group(1).replace("(", "（").replace(")", "）").strip()

    if m := _RE_NUMBER.search(text):
        invoice.invoice_number = m.group(1)

    invoice.issue_date = _parse_date_flexible(text)

    # 购买方 / 销售方分区：两个"纳税人识别号"必须归属到正确的一方
    seller_idx = text.find("销售方名称")
    buyer_block = text[:seller_idx] if seller_idx != -1 else text
    seller_block = text[seller_idx:] if seller_idx != -1 else ""

    if m := _RE_BUYER.search(buyer_block):
        invoice.buyer_name = m.group(1).strip()
    if m := _RE_SELLER.search(seller_block):
        invoice.seller_name = m.group(1).strip()

    buyer_ids = _RE_TAXID.findall(buyer_block)
    if buyer_ids:
        invoice.buyer_tax_id = buyer_ids[-1]
    seller_ids = _RE_TAXID.findall(seller_block)
    if seller_ids:
        invoice.seller_tax_id = seller_ids[0]

    if m := _RE_ITEM_LINE.search(text):
        invoice.item_name = m.group(1).strip()
        invoice.amount = _safe_money(m.group(4))
        invoice.tax_rate = m.group(5)
        invoice.tax_amount = _safe_money(m.group(6))

    if m := _RE_TOTAL_SMALL.search(text):
        invoice.total = _safe_money(m.group(1))
    if m := (_RE_TOTAL_WORDS.search(text) or _RE_TOTAL_WORDS_ALT.search(text)):
        invoice.total_in_words = m.group(1).strip()

    if m := _RE_REMARK.search(text):
        invoice.remark = m.group(1).strip()

    # 兜底：有些版式把价税合计写成"合计金额"或只有一个总额
    if invoice.total is None and invoice.amount is not None and invoice.tax_amount is not None:
        invoice.total = parse_money(invoice.amount + invoice.tax_amount)

    if not invoice.invoice_number and invoice.total is None:
        raise ExtractionError("PDF 文本层里找不到发票关键字段（号码/金额），可能版式不受支持")

    return invoice


def _safe_money(value: str):
    try:
        return parse_money(value)
    except ValueError:
        return None


def _parse_date_flexible(text: str) -> date | None:
    """先试「YYYY年MM月DD日」，再试「YYYY-MM-DD」。"""
    if m := _RE_DATE.search(text):
        y, mo, d = (int(g) for g in m.groups())
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    if m := _RE_DATE_ALT.search(text):
        y, mo, d = (int(g) for g in m.groups())
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    return None
