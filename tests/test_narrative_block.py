"""叙述证据块的可序列化性
========================

``narrative_evidence_block()`` 的输出会进提示词，所以它**不能抛异常** ——
它抛了，``audit._make_narrative`` 的 try/except 会静默换成模板，
表面看一切正常，实际是"这类单据的审核意见永远是模板文字，模型一次没被用上"。

这个缺陷不是想出来的，是实测撞出来的：限额类规则（R006 / R007 / R013）
的 evidence 里带 ``Decimal``（金额来自 ``parse_money``），``json.dumps`` 不认。

所以这里钉两类东西：

1. ``Decimal`` 在 evidence 里必须能渲染（不能抛，且要出现在文本里）；
2. 真实样本跑出来的结果，其证据块必须可渲染 —— 防止将来某条新规则又往
   evidence 里塞进不可序列化的对象。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal


from finance.models import (
    AuditFinding,
    AuditResult,
    Invoice,
    ReimbursementRequest,
    Severity,
)


def _result_with(findings: list[AuditFinding]) -> AuditResult:
    return AuditResult(
        invoice=Invoice(invoice_number="24312000000000000001", total=Decimal("600.00")),
        request=ReimbursementRequest(
            applicant="张三",
            department="技术部",
            expense_type="住宿费",
            amount=Decimal("600.00"),
            submit_date=date(2026, 9, 18),
        ),
        audit_id="test0001",
        findings=findings,
    )


def test_evidence_里的_Decimal_不影响渲染() -> None:
    finding = AuditFinding(
        rule_id="R007",
        clause="第 4.2 条",
        title="住宿费不超过城市分级限额",
        severity=Severity.FAIL,
        message="上海（一线城市）住宿 601.00 元/晚 > 限额 600.00 元/晚",
        evidence={
            "field": "每晚住宿费",
            "actual": Decimal("601.00"),          # ← 就是这个让 json.dumps 崩过
            "expected": Decimal("600.00"),
        },
    )
    block = _result_with([finding]).narrative_evidence_block()

    assert "R007" in block
    assert "601.00" in block and "600.00" in block, "金额要真的出现在提示词里，否则模型只能编"


def test_不可序列化对象也不该炸() -> None:
    """date 之类的对象同样不该把证据块搞崩（叙述层永远不该阻塞审核）。"""
    finding = AuditFinding(
        rule_id="R003",
        clause="第 3.2 条",
        title="报销时限",
        severity=Severity.FAIL,
        message="开票日距今 90 天",
        evidence={"issue_date": date(2026, 6, 20)},
    )
    block = _result_with([finding]).narrative_evidence_block()
    assert "2026-06-20" in block
