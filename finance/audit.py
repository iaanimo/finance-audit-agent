"""审核编排 —— 唯一的状态变更入口
====================================

整个 finance 包里，**只有这个模块能改变审核单的状态**。别处（工具层、Web 层）
都只能调用它、或者只读地看它返回的结果。这条约束由 tests 里的不变量断言守住。

为什么要这么严：状态变更意味着"这张单子往前走了一步"。如果到处都能改状态，
演示时你就答不上来"这一步是谁触发的"。收敛到一个入口，责任才清楚。

流程（六态状态机，每一态都对应真实发生的一件事）
------------------------------------------------

    抽取 extract          -> extracted
    静态校验                -> validated
    查重（需要台账）        -> duplicate_checked
    预算校验（需要预算表）  -> budget_checked
    生成凭证草稿 + 试算平衡 -> draft_created
    等待人工决定            -> pending_review
    人工决定                -> approved / rejected

前四态是**系统**走完的，第五态到第六态的跨越**只有人能做**——
这就是"人工闸门"，也是这个项目和"全自动 AI 审批"的分界线。

每一步都往 ``.log.jsonl`` 追加一条审计事件（只追加，不改）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .guard import guard_narrative
from .models import (
    AuditResult,
    AuditState,
    Decision,
    Invoice,
    NarrativeSource,
    ReimbursementRequest,
    Severity,
    SuggestedStatus,
)
from .policy import PolicyBundle, load_policy_bundle
from .rules import aggregate, evaluate, summarize_findings
from .store import AuditStore
from .voucher import VoucherError, build_voucher

# --------------------------------------------------------------------------
# 阶段划分
# --------------------------------------------------------------------------
#
# 规则不是一把跑完的 —— 分阶段是为了让状态机有实际含义：
# 查重必须等台账接进来，预算必须等预算表接进来，凭证平衡必须等凭证生成。
# tests 里有一条断言：rules.yaml 里的每条规则必须恰好属于一个阶段，
# 防止以后新增规则时忘了归类。

STAGE_VALIDATE = [
    "R001", "R002", "R003", "R005", "R006",
    "R007", "R008", "R009", "R010", "R011",
    "R015", "R016", "R017",
]
STAGE_HISTORY = ["R004", "R012"]
STAGE_BUDGET = ["R014"]
STAGE_VOUCHER = ["R013"]

STAGE_BY_RULE: dict[str, str] = {
    **{r: "validate" for r in STAGE_VALIDATE},
    **{r: "history" for r in STAGE_HISTORY},
    **{r: "budget" for r in STAGE_BUDGET},
    **{r: "voucher" for r in STAGE_VOUCHER},
}

NARRATIVE_TIMEOUT_SECONDS = 25


class OverrideReasonRequired(Exception):
    """人工决定与系统建议相反时必须书面说明理由（制度 2.3）。"""


class AuditError(Exception):
    """审核流程本身出错（不是业务判定，是流程故障）。"""


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


async def run_audit(
    invoice_path: str | Path,
    request: ReimbursementRequest,
    *,
    store: AuditStore,
    policy: PolicyBundle | None = None,
    use_vision: bool = False,
    vision_timeout: int = 20,
    narrative_llm: Any = "auto",
) -> AuditResult:
    """跑完一遍审核，返回停在 ``pending_review`` 的结果并落盘。

    **注意这里没有"批准"这一步** —— 系统只能把单子推到"等人工决定"。
    批准或驳回只有 :func:`decide` 能做，而它必须由人调用。

    :param narrative_llm: ``"auto"`` 用默认模型；``None`` 完全不用模型（模板叙述）。
    """
    from .extractor import extract

    policy = policy or load_policy_bundle()

    # ---- 状态 1：抽取 ----
    invoice: Invoice = extract(
        invoice_path, use_vision=use_vision, vision_timeout=vision_timeout
    )
    result = AuditResult(
        invoice=invoice, request=request, state=AuditState.EXTRACTED
    )
    store.append_log(
        result.audit_id,
        {
            "event": "extracted",
            "method": invoice.extraction_method,
            "source_file": invoice.source_file,
            "invoice_number": invoice.invoice_number,
        },
    )

    history = store.history_view()

    # ---- 状态 2：静态校验 ----
    result.findings.extend(
        evaluate(invoice, request, policy, history=history, only=STAGE_VALIDATE)
    )
    result.state = AuditState.VALIDATED
    _log_stage(store, result, "validated", STAGE_VALIDATE)

    # ---- 状态 3：查重 ----
    result.findings.extend(
        evaluate(invoice, request, policy, history=history, only=STAGE_HISTORY)
    )
    result.state = AuditState.DUPLICATE_CHECKED
    _log_stage(store, result, "duplicate_checked", STAGE_HISTORY)

    # ---- 状态 4：预算 ----
    result.findings.extend(
        evaluate(invoice, request, policy, history=history, only=STAGE_BUDGET)
    )
    result.state = AuditState.BUDGET_CHECKED
    _log_stage(store, result, "budget_checked", STAGE_BUDGET)

    # ---- 状态 5：凭证草稿 + 借贷平衡 ----
    #
    # 只有在"没有事实性违规"时才生成凭证。被驳回的单子不该产生凭证草稿
    # （那会让会计以为有话可入账）。若人后来推翻了驳回，decide() 会补生成。
    blocking = [f for f in result.findings if f.severity is Severity.FAIL]
    if not blocking:
        _try_build_voucher(result, policy)
    result.findings.extend(
        evaluate(
            invoice, request, policy,
            history=history, voucher=result.voucher, only=STAGE_VOUCHER,
        )
    )
    result.state = AuditState.DRAFT_CREATED
    _log_stage(store, result, "draft_created", STAGE_VOUCHER, extra={
        "voucher_created": result.voucher is not None,
        "balanced": result.voucher.balanced if result.voucher else None,
    })

    # ---- 状态 6：待人工决定 ----
    result.suggested_status = aggregate(result.findings)
    template = summarize_findings(result.findings)
    result.narrative, result.narrative_source = await _make_narrative(
        result, template, narrative_llm
    )
    result.state = AuditState.PENDING_REVIEW

    store.append_log(
        result.audit_id,
        {
            "event": "submitted",
            "suggested_status": result.suggested_status.value,
            "narrative_source": result.narrative_source.value,
            "blocking": [f.rule_id for f in result.findings if f.severity is Severity.FAIL],
            "warnings": [f.rule_id for f in result.findings if f.severity is Severity.WARN],
        },
    )
    store.save(result)
    return result


def decide(
    result: AuditResult,
    decision: Decision,
    operator: str,
    *,
    store: AuditStore,
    override_reason: str = "",
    policy: PolicyBundle | None = None,
) -> AuditResult:
    """人工决定 —— **系统之外唯一能推进状态的入口**。

    制度 2.3：人工有权采纳或推翻系统建议，但**推翻必须书面说明理由**。
    这里做服务端强制，且记进只追加的审计日志 ——
    *系统有意见，人有权力，但权力留下痕迹。*

    :raises OverrideReasonRequired: 推翻系统建议却没给理由
    """
    if result.decision is not None:
        raise AuditError(f"审核单 {result.audit_id} 已由 {result.operator} 做过决定")

    if result.is_overridden_now(decision) and not override_reason.strip():
        raise OverrideReasonRequired(
            f"系统建议为 {result.suggested_status.value}，"
            f"你的决定是 {decision.value}，属于推翻系统建议，必须填写书面理由"
        )

    # 推翻"驳回"而批准 —— 补一张凭证草稿，否则后面无账可入
    if decision is Decision.APPROVED and result.voucher is None:
        policy = policy or load_policy_bundle()
        _try_build_voucher(result, policy)

    result.decision = decision
    result.operator = operator
    result.override_reason = override_reason.strip()
    result.decided_at = datetime.now(timezone.utc)
    result.state = (
        AuditState.APPROVED if decision is Decision.APPROVED else AuditState.REJECTED
    )

    store.append_log(
        result.audit_id,
        {
            "event": "decided",
            "suggested_status": result.suggested_status.value,
            "decision": decision.value,
            "operator": operator,
            "overridden": result.is_overridden,
            "override_reason": result.override_reason,
        },
    )
    store.save(result)
    return result


# --------------------------------------------------------------------------
# 内部
# --------------------------------------------------------------------------


def _log_stage(
    store: AuditStore,
    result: AuditResult,
    stage: str,
    rule_ids: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """记录一次阶段推进，附上该阶段跑出的结论分布。"""
    stage_findings = [f for f in result.findings if f.rule_id in rule_ids]
    event = {
        "event": "stage",
        "stage": stage,
        "rules": rule_ids,
        "result": {f.rule_id: f.severity.value for f in stage_findings},
    }
    event.update(extra or {})
    store.append_log(result.audit_id, event)


def _try_build_voucher(result: AuditResult, policy: PolicyBundle) -> None:
    """尽力生成凭证。科目归类不了就跳过 —— 由 R011/R013 的结论去提示人工。"""
    try:
        result.voucher = build_voucher(result.invoice, result.request, policy)
    except VoucherError:
        result.voucher = None


async def _make_narrative(
    result: AuditResult, template: str, narrative_llm: Any
) -> tuple[str, NarrativeSource]:
    """生成审核意见叙述。**任何失败都不影响结论，只影响文字。**

    LLM 写完必须过 :func:`finance.guard.guard_narrative` 护栏：
    叙述里每个数字都得能在 findings 里找到，否则整段换成模板。
    """
    if narrative_llm is None:
        return template, NarrativeSource.TEMPLATE

    try:
        if narrative_llm == "auto":
            from core.llm_factory import create_llm

            narrative_llm = create_llm(temperature=0.0)

        prompt = _build_narrative_prompt(result)
        response = await asyncio.wait_for(
            narrative_llm.ainvoke(prompt), timeout=NARRATIVE_TIMEOUT_SECONDS
        )
        text = getattr(response, "content", "") or ""
        return guard_narrative(text, result.findings, template)
    except Exception:  # noqa: BLE001 —— 叙述失败绝不能让审核失败
        return template, NarrativeSource.TEMPLATE


def _build_narrative_prompt(result: AuditResult) -> str:
    """构造叙述提示词。**明确的数字约束**是护栏的第一道防线。"""
    lines = [
        "你是企业财务共享中心的审核意见撰写助手。",
        "下面是一份报销单的系统审核结论。请写一段 100 字以内的中文审核意见。",
        "",
        "硬性要求：",
        "1. 只能引用下面出现过的数字，不得计算、不得推测、不得引入任何新数字。",
        "2. 不得改变任何一条规则的判定结论，你只负责把结论说清楚。",
        "3. 不要写「作为AI」之类的自我介绍，直接给意见。",
        "4. 如果有驳回项，要明确指出命中哪条制度条款。",
        "",
        "=== 单据信息 ===",
        f"申请人：{result.request.applicant}　部门：{result.request.department}",
        f"费用类型：{result.request.expense_type}　申请金额：{result.request.amount}",
        f"发票价税合计：{result.invoice.total}　销售方：{result.invoice.seller_name}",
        "",
        "=== 逐条判定 ===",
        result.narrative_evidence_block(),
        "",
        f"=== 系统建议 ==={result.suggested_status.value}",
    ]
    return "\n".join(lines)
