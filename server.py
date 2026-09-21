"""财务报销审核受控 Agent —— Web 服务
=======================================

启动::

    ./.venv/Scripts/python.exe server.py
    # 浏览器打开 http://127.0.0.1:8000/audit

**这条链路刻意不经过任何 agent 循环，也不给模型任何工具。**

审核走的是一条**固定的、可复现的管线**（``finance/audit.py::run_audit``）：
LLM 只在这条管线的两端出现 —— 入口的字段抽取、出口的叙述润色；
中间 17 条规则的判定完全由 Python 完成。

这不是"没用上 agent 能力"，这是**设计**：财务审核的结论必须可复现，
不能交给一个会自由发挥的循环。

**一个实测过的坑**：本机（以及绝大多数精简部署）没有装 ``python-multipart``，
所以 **不能用** ``UploadFile`` / ``File()`` / ``Form()`` —— 一写就 500。
发票文件走 base64 塞进 JSON 传输。
"""

from __future__ import annotations

import argparse
import base64
import binascii
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from finance import Decision, ReimbursementRequest, parse_money
from finance.audit import AuditError, OverrideReasonRequired, decide, run_audit
from finance.store import AuditStore
from tools.file_ops import resolve_data_path

HERE = Path(__file__).resolve().parent
LOGS_DIR = HERE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "finance-audit.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("finance-audit")

app = FastAPI(title="财务报销审核受控 Agent", docs_url="/docs")

AUDIT_STORE = AuditStore()
SAMPLE_DIR = HERE / "finance" / "samples"
UPLOAD_DIR_NAME = "uploads"
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024

_ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class AuditRunRequest(BaseModel):
    filename: str = Field(default="", description="原始文件名，用于留痕与后缀判断")
    content_b64: str = Field(default="", description="发票文件的 base64 内容")
    use_vision: bool = Field(
        default=False, description="PDF 无文本层时是否允许调视觉模型兜底"
    )
    request: dict = Field(default_factory=dict, description="报销申请单字段")


class AuditDecideRequest(BaseModel):
    decision: str = Field(description="APPROVED 或 REJECTED")
    operator: str = Field(default="", description="操作人")
    override_reason: str = Field(
        default="", description="推翻系统建议时的书面理由（制度 2.3 要求必填）"
    )


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    """根路径直接进审核台。"""
    return RedirectResponse(url="/audit")


@app.get("/audit")
async def audit_page():
    page = HERE / "static" / "audit.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="audit.html 不存在")
    return FileResponse(page)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 样本票（演示用）
# ---------------------------------------------------------------------------


@app.get("/api/audit/samples")
async def audit_samples():
    """列出内置合成样本票，供演示时一键选用。"""
    manifest_path = SAMPLE_DIR / "samples.yaml"
    if not manifest_path.is_file():
        return {"samples": []}
    import yaml

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return {
        "samples": [
            {
                "key": key,
                "title": spec.get("title", ""),
                "expects": spec.get("expects", ""),
                "pdf": spec.get("pdf", ""),
                "request": spec.get("request", {}),
            }
            for key, spec in manifest.items()
        ]
    }


@app.get("/api/audit/sample/{key}")
async def audit_sample_file(key: str):
    """取一张样本票的 PDF 原文（前端转 base64 后回传，模拟"上传"）。"""
    safe = "".join(ch for ch in key if ch.isalnum() or ch in "-_")
    path = SAMPLE_DIR / "pdf" / f"{safe}.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"样本不存在：{key}")
    return FileResponse(path, media_type="application/pdf", filename=f"{safe}.pdf")


# ---------------------------------------------------------------------------
# 审核
# ---------------------------------------------------------------------------


@app.get("/api/audit/list")
async def audit_list():
    return {"audits": AUDIT_STORE.list_audits()}


@app.post("/api/audit/run")
async def audit_run(req: AuditRunRequest):
    """提交一张发票跑审核。

    **系统只能把单子推到"待人工决定"。** 这个端点里没有任何一行能批准报销。
    """
    if not req.content_b64:
        raise HTTPException(status_code=400, detail="发票内容为空。")

    try:
        blob = base64.b64decode(req.content_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="发票内容不是合法的 base64。")

    if len(blob) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="发票文件过大（上限 8MB）。")

    suffix = Path(req.filename).suffix.lower() or ".pdf"
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix}")

    # ⚠️ 顺序要紧：**先校验申请单，再落盘**。
    # 反过来的话，一张缺字段的废单会先把文件写进 data/uploads/ 再被 400 拒掉，
    # 留下一个没人引用的孤儿文件。
    try:
        request = _build_request(req.request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"报销申请单字段有误：{exc}")

    target = resolve_data_path(f"{UPLOAD_DIR_NAME}/{uuid.uuid4().hex[:10]}{suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)

    try:
        result = await run_audit(
            target, request, store=AUDIT_STORE, use_vision=req.use_vision
        )
    except Exception as exc:  # noqa: BLE001 —— 抽取失败给人话提示，不是 500
        logger.warning("audit run failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"发票处理失败：{exc}")

    logger.info(
        "audit=%s suggested=%s narrative_source=%s",
        result.audit_id,
        result.suggested_status.value,
        result.narrative_source.value,
    )
    return result.to_json_dict()


@app.get("/api/audit/{audit_id}")
async def audit_detail(audit_id: str):
    result = AUDIT_STORE.load(audit_id)
    if result is None:
        raise HTTPException(status_code=404, detail="审核单不存在。")
    return result.to_json_dict()


@app.get("/api/audit/{audit_id}/log")
async def audit_log(audit_id: str):
    """审计轨迹。只追加，不可修改 —— 页面上的"可追溯"指的就是这个。"""
    return {"events": AUDIT_STORE.read_log(audit_id)}


@app.post("/api/audit/{audit_id}/decide")
async def audit_decide(audit_id: str, req: AuditDecideRequest):
    """人工决定。**全系统唯一能批准报销的入口。**

    与系统建议相反而未填理由 -> 400（制度 2.3 的服务端强制）。
    """
    result = AUDIT_STORE.load(audit_id)
    if result is None:
        raise HTTPException(status_code=404, detail="审核单不存在。")

    try:
        decision = Decision(req.decision.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="decision 只能是 APPROVED 或 REJECTED。")

    if not req.operator.strip():
        raise HTTPException(status_code=400, detail="必须填写操作人。")

    try:
        decide(
            result,
            decision,
            req.operator.strip(),
            store=AUDIT_STORE,
            override_reason=req.override_reason,
        )
    except OverrideReasonRequired as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except AuditError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return result.to_json_dict()


@app.post("/api/audit/reset")
async def audit_reset():
    """清空演示数据。

    ⚠️ **演示前必须调用一次。** 排练时跑过的单子会留在查重台账里，
    正式演示再传同一张 S01，R004 会命中"重复报销"，开场全绿基线当场翻车。
    这是本项目最容易踩的一个演示坑，页面上有对应按钮。
    """
    removed = AUDIT_STORE.clear()
    upload_dir = resolve_data_path(UPLOAD_DIR_NAME)
    if upload_dir.is_dir():
        for f in upload_dir.iterdir():
            if f.is_file():
                f.unlink()
    logger.info("audit demo reset: %s files removed", removed)
    return {"removed": removed}


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------


def _build_request(spec: dict) -> ReimbursementRequest:
    """把前端传来的 dict 转成报销申请单。缺字段给人话报错，不是 500。

    **必填项在这里拦死，不能只靠页面拦。** 审核台的入口是公开接口，
    谁都能绕过表单直接 POST 一个空对象过来；放过去的话会生成一张
    「申请人空、金额 0」的废单，真的落进查重台账。

    与页面校验的分工：页面拦是为了**立刻**告诉人缺什么，
    这里拦才是**真正**拦得住的那一道。
    """
    from datetime import date

    if not spec:
        raise ValueError("缺少报销申请单数据")

    # 提交日期**不设默认值**。曾经缺省取服务器当天 —— 那等于系统替申请人
    # 编了一个申报日期，而 R003（开票日距提交日 <= 60 天、不得跨年）正是
    # 拿它当判定基准的。申报日期是事实，只能由申请人显式给出。
    submit_raw = spec.get("submit_date")
    if not submit_raw:
        raise ValueError("提交日期未填")
    try:
        submit_date = date.fromisoformat(str(submit_raw))
    except ValueError:
        raise ValueError(f"提交日期格式应为 YYYY-MM-DD，收到 {submit_raw!r}")

    try:
        amount = parse_money(spec.get("amount", 0))
    except ValueError:
        raise ValueError(f"申请金额无法解析：{spec.get('amount')!r}")

    applicant = str(spec.get("applicant") or "").strip()
    department = str(spec.get("department") or "").strip()
    expense_type = str(spec.get("expense_type") or "").strip()

    missing = [
        label
        for label, value in (
            ("申请人", applicant),
            ("所属部门", department),
            ("费用类型", expense_type),
        )
        if not value
    ]
    if missing:
        raise ValueError("以下必填项为空：" + "、".join(missing))

    # 金额 0 不是"小金额"，是**没填**。放过去只会得到一张
    # R010「申请金额 0.00 ≠ 发票 X」的废单 —— 没有财务风险，但污染台账。
    if amount <= 0:
        raise ValueError(f"申请金额必须大于 0，收到 {amount}")

    nights = spec.get("nights")
    headcount = spec.get("headcount")

    return ReimbursementRequest(
        applicant=applicant,
        department=department,
        expense_type=expense_type,
        amount=amount,
        reason=str(spec.get("reason") or "").strip(),
        submit_date=submit_date,
        city=str(spec.get("city") or "").strip(),
        nights=int(nights) if nights not in (None, "") else None,
        headcount=int(headcount) if headcount not in (None, "") else None,
        has_itemized_list=bool(spec.get("has_itemized_list")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="财务报销审核受控 Agent")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址（0.0.0.0 可局域网访问）")
    parser.add_argument("--port", type=int, default=8000, help="端口")
    args = parser.parse_args()

    import uvicorn

    settings = get_settings()
    logger.info("审核台: http://%s:%s/audit", args.host, args.port)
    if not settings.api_key:
        logger.warning(
            "未配置 API Key —— 审核功能不受影响，审核意见会退化为模板文字。"
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
