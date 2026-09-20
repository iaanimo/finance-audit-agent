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
   因为演示时不该依赖网络。

两个必须绕开的坑（都是实测出来的）
----------------------------------
- ``describe_image.describe`` 内部会用 ``raise SystemExit`` 报错。``SystemExit``
  继承 ``BaseException`` 而不是 ``Exception``，所以 ``except Exception`` **兜不住**，
  会直接把 uvicorn 进程干掉。本模块一律用 ``except BaseException`` 接住。
- ``describe_image`` 的 ``max_tokens`` 是硬编码的 1024，十几个字段的中文 JSON
  会被截断。本模块显式传更大的值，并且做"截断后二次解析"容错。
"""

from __future__ import annotations

import json
import re
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
            # PDF 文本层读不出来（扫描件？）—— 转视觉兜底
            try:
                return extract_from_image(p, timeout=vision_timeout)
            except ExtractionError:
                raise pdf_exc

    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
        return extract_from_image(p, timeout=vision_timeout)

    raise ExtractionError(f"不支持的文件类型: {suffix}（支持 pdf 与常见图片格式）")


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
        raise ExtractionError("PDF 没有可提取的文本层（可能是扫描件，请改用图片路径）")

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

    **注意** ``except BaseException``：``describe_image`` 内部用 ``SystemExit``
    报错，它不是 ``Exception``，普通兜底接不住，会把 Web 进程一起带走。
    """
    p = Path(path)
    try:
        from describe_image import describe
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("无法导入 describe_image 模块") from exc

    try:
        raw = describe(str(p), prompt=VISION_PROMPT, timeout=timeout, max_tokens=2048)
    except BaseException as exc:  # noqa: BLE001 —— 刻意接住 SystemExit
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
