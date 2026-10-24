"""推翻统计：归因口径与空态
============================

这个脚本不是 RLHF —— 它只输出报告，不改规则。测试守三件事：

1. 推翻"驳回建议"时，归因到命中的驳回规则；
2. 推翻"通过建议"时**不**归因到任何规则（人工自由裁量，不是规则的错）；
3. 没有数据时给出可读的空态，而不是一个空表。
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import pytest

from finance.audit import decide, run_audit
from finance.models import Decision, ReimbursementRequest
from finance.store import AuditStore
from scripts.measure_overrides import collect, render, summarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = PROJECT_ROOT / "finance" / "samples" / "pdf"


def make_request(**over) -> ReimbursementRequest:
    base = dict(
        applicant="张三",
        department="技术部",
        expense_type="住宿费",
        amount="1650.00",
        reason="上海客户现场支持",
        submit_date=date(2026, 9, 18),
        city="上海",
        nights=3,
    )
    base.update(over)
    return ReimbursementRequest(**base)


def _run(store, pdf_name: str, **req_over):
    pdf = SAMPLES / pdf_name
    if not pdf.is_file():
        pytest.skip(f"样本缺失：{pdf_name}")

    async def go():
        return await run_audit(pdf, make_request(**req_over), store=store, narrative_llm=None)

    return asyncio.run(go())


def test_override_of_rejection_is_attributed_to_blocking_rule(tmp_path):
    store = AuditStore(base_dir=tmp_path / "audits")
    result = _run(store, "S02_hotel_over_limit.pdf", amount="2400.00")
    decide(
        result,
        Decision.APPROVED,
        "李会计",
        store=store,
        override_reason="客户临时改期，超标部分员工自付，已电话确认",
    )

    rows = collect(store)
    assert len(rows) == 1
    assert rows[0]["overridden"] is True
    assert [f["rule_id"] for f in rows[0]["rules"]] == ["R007"]  # 一线城市限额

    summary = summarize(rows)
    assert summary["overridden"] == 1
    assert summary["by_rule"] == {"R007": 1}
    assert summary["unattributed"] == 0
    assert summary["by_operator"] == {"李会计": 1}


def test_override_of_approval_has_no_rule_to_blame(tmp_path):
    store = AuditStore(base_dir=tmp_path / "audits")
    result = _run(store, "S01_hotel_ok.pdf")  # 系统建议 APPROVED
    decide(
        result,
        Decision.REJECTED,
        "李会计",
        store=store,
        override_reason="该员工本月已报过一次同类费用，业务上重复",
    )

    summary = summarize(collect(store))
    assert summary["overridden"] == 1
    assert summary["by_rule"] == {}
    assert summary["unattributed"] == 1  # 系统当时没意见，不该怪某条规则


def test_normal_decision_is_not_counted_as_override(tmp_path):
    store = AuditStore(base_dir=tmp_path / "audits")
    result = _run(store, "S01_hotel_ok.pdf")
    decide(result, Decision.APPROVED, "李会计", store=store)

    summary = summarize(collect(store))
    assert summary["decided"] == 1
    assert summary["overridden"] == 0
    assert summary["rate"] == 0.0


def test_empty_state_is_readable():
    text = render(summarize([]))
    assert "暂无已做决定的单据" in text
