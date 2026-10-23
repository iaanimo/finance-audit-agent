"""ERP 无状态审核接口（G3-4b 审核桥的 nono 侧壳）
==================================================

对 ERP 暴露两口（契约方：ERP 仓库 `backend/app/services/audit_bridge.py`）：

- ``POST /audit`` —— 无状态送审：DTO 进、结论出，**不落 store、不建 audit_id**；
- ``GET /version`` —— 规则版本（`rule_version` ↔ `code_ref`），供 ERP 登记与巡检。

设计要点（与 Web UI 管线 `server.py::api/audit/run` 的区别）：

1. **无状态**：直调 `finance.rules.evaluate`（19 条规则判定核心），刻意不走
   `finance.audit.run_audit`——那条管线绑发票抽取 + `AuditStore` 落盘，是
   Web UI 的形态；集成场景单据进结论出，审核数据只过内存（合规：不落副本）。
2. **判定零 LLM**：本壳只做规则判定，叙述/凭证等 LLM 两端不进集成面——
   结论可复现是 ERP 桥的立论。
3. **响应必带 `rule_version`**：ERP 侧把"缺版本"视为协议异常降级（K04），
   所以 200/400 都带；规则集（`finance/rules.py` / `config/policies`）变更时
   **升 `RULE_VERSION`**——历史结论可按当时版本复算（审核可复算口径）。
4. **severity 映射**：nono 三态 PASS/WARN/FAIL → ERP 三档 info/warn/block；
   `applicable=False`（规则不适用）→ info；规则执行异常（`error`）→ warn。

状态码语义（契约方约定）：4xx = DTO 问题（ERP 记 degraded 待补审）；
本壳是纯函数面，不会 5xx（单条规则异常已被 `evaluate` 降级为 WARN）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .models import AuditFinding, Invoice, ReimbursementRequest, Severity
from .policy import load_policy_bundle
from .rules import evaluate

#: 规则集版本——rules.py / config/policies 变更时升版（历史结论按当时版本复算）
RULE_VERSION = "v1.0.0"
CODE_REF = f"nono/finance/rules@{RULE_VERSION}"
DTO_VERSION = "1"

router = APIRouter(tags=["erp"])

_SEVERITY_MAP: dict[Severity, str] = {
    Severity.PASS: "info",
    Severity.WARN: "warn",
    Severity.FAIL: "block",
}


# ---------------------------------------------------------------------------
# ERP DTO v1（与 audit_bridge.build_audit_dto 逐字段对齐）
# ---------------------------------------------------------------------------


class ErpPartner(BaseModel):
    name: str = ""
    partner_type: str = ""


class ErpLine(BaseModel):
    material_code: str = ""
    quantity: float | None = None
    price: float | None = None


class ErpDocument(BaseModel):
    doc_type: str = ""
    doc_no: str = ""
    doc_date: str = ""
    total_amount: str | float = ""
    currency_code: str = "CNY"
    # None / 缺省均接受（真连实测：ERP 无对手方时显式送 null——接收方宽容）
    partner: ErpPartner | None = None
    lines: list[ErpLine] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class ErpAuditRequest(BaseModel):
    dto_version: str = DTO_VERSION
    document: ErpDocument


def _bad_request(msg: str) -> HTTPException:
    """DTO 问题 → 400，**响应体仍带 rule_version**（契约：任何响应必带）。"""
    return HTTPException(
        status_code=400,
        detail={"message": msg, "rule_version": RULE_VERSION},
    )


def _to_models(doc: ErpDocument) -> tuple[Invoice, ReimbursementRequest]:
    """ERP 单据 DTO → nono 审核模型（无状态适配，不落库）。

    映射是"尽力适配"：发票类规则（购买方/税率等）在非发票单据上会给出
    N/A 或提示——**如实反映规则判定**，不替规则编结论。
    """
    try:
        doc_date = datetime.strptime(doc.doc_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise _bad_request(f"doc_date 不合法（应为 YYYY-MM-DD）：{doc.doc_date!r}") from exc
    try:
        amount = Decimal(str(doc.total_amount))
    except (InvalidOperation, ValueError) as exc:
        raise _bad_request(f"total_amount 不合法：{doc.total_amount!r}") from exc

    item_name = ",".join(line.material_code for line in doc.lines if line.material_code)
    partner_name = doc.partner.name if doc.partner else ""
    invoice = Invoice(
        invoice_number=doc.doc_no,
        issue_date=doc_date,
        seller_name=partner_name,
        buyer_name=str(doc.extra.get("buyer_name", "")),
        item_name=item_name or doc.doc_type,
        total=amount,
        raw_text="",  # 集成面无原文——证据链根在 ERP 侧单据
    )
    request = ReimbursementRequest(
        applicant=str(doc.extra.get("applicant") or partner_name),
        department=str(doc.extra.get("department", "")),
        expense_type=str(doc.extra.get("expense_type") or doc.doc_type),
        amount=amount,
        reason=str(doc.extra.get("reason") or doc.doc_no),
        submit_date=doc_date,
        note=str(doc.extra.get("note", "")),
    )
    return invoice, request


def _map_finding(f: AuditFinding) -> dict[str, Any]:
    if f.error is not None:
        severity = "warn"
    elif not f.applicable:
        severity = "info"
    else:
        severity = _SEVERITY_MAP.get(f.severity, "warn")
    return {
        "rule_code": f.rule_id,
        "severity": severity,
        "message": f.message,
        "ref": f.clause or None,
    }


# ---------------------------------------------------------------------------
# 契约两口
# ---------------------------------------------------------------------------


@router.get("/version")
async def erp_version() -> dict[str, str]:
    """规则版本（ERP 侧 ai_rule_set 登记 / 巡检用）。"""
    return {"rule_version": RULE_VERSION, "code_ref": CODE_REF}


@router.post("/audit")
async def erp_audit(req: ErpAuditRequest) -> dict[str, Any]:
    """无状态送审：DTO 进、结论出（不落 store、不建 audit_id、零 LLM）。"""
    if req.dto_version != DTO_VERSION:
        raise _bad_request(
            f"dto_version 不支持：{req.dto_version!r}（当前支持 {DTO_VERSION!r}）"
        )
    invoice, request = _to_models(req.document)
    findings = evaluate(invoice, request, load_policy_bundle())
    return {
        "rule_version": RULE_VERSION,
        "code_ref": CODE_REF,
        "findings": [_map_finding(f) for f in findings],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
