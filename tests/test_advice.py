"""补正建议的测试
================

``finance/advice.py`` 有三条硬约束，这里逐条钉住：

1. **算得对**：建议里的数字来自 evidence（限额、金额、人数），不是编的；
2. **不教规避**：任何一条建议都不许出现"拆单 / 拆分 / 拆成两张"这类出主意；
3. **只读**：它不改变判定、不改变状态，只把结论换个说法。
"""

from __future__ import annotations


import pytest

from finance.advice import suggest_advice, suggest_dispositions
from finance.models import AuditFinding, Severity


def _f(rule_id: str, **ev) -> AuditFinding:
    return AuditFinding(
        rule_id=rule_id,
        clause="4.2",
        title="测试用规则",
        severity=Severity.FAIL,
        message="测试消息",
        evidence=ev,
    )


def test_住宿超标给出可执行的降标金额() -> None:
    f = _f("R007", actual=800.0, expected="<= 600.00", nights=3, invoice_total=2400.0)
    (item,) = suggest_advice([f])
    assert item["kind"] == "可补正"
    assert "600.00" in item["text"]
    assert "1,800.00" in item["text"], "3 晚降到限额后的合计要算出来"
    assert "2,400.00" in item["text"]


def test_申请金额不符直接给出应填的数() -> None:
    f = _f("R010", claimed_amount=500.0, invoice_total=550.0, actual=500.0, expected=550.0)
    (item,) = suggest_advice([f])
    assert "500.00" in item["text"] and "550.00" in item["text"]
    assert "改为" in item["text"]


def test_餐饮超标给出金额上限而不是教人凑人数() -> None:
    f = _f("R006", actual=4, invoice_total=700.0, expected="<= 150.00")
    (item,) = suggest_advice([f])
    assert "600.00" in item["text"], "4 人 × 150 = 600.00 元上限"
    assert "不得自行补人数凑标准" in item["text"]


def test_餐饮缺人数时算出最少需要几人() -> None:
    f = _f("R006", actual=None, invoice_total=600.0, expected=">= 1")
    (item,) = suggest_advice([f])
    assert item["kind"] == "补充材料"
    assert "4 人" in item["text"] or "至少 4" in item["text"]


def test_超期只能走人工或重开票() -> None:
    f = _f("R003", issue_date="2026-06-20", submit_date="2026-09-18", age_days=90,
           expected="<= 60 天且同一年度")
    (item,) = suggest_advice([f])
    assert item["kind"] == "须人工/特批"
    assert "90" in item["text"]


@pytest.mark.parametrize(
    "rule_id,ev",
    [
        ("R007", {"actual": 800.0, "expected": "<= 600.00", "nights": 3}),
        ("R006", {"actual": 4, "invoice_total": 700.0, "expected": "<= 150.00"}),
        ("R008", {"actual": 2690.0, "expected": "<= 2,000.00"}),
        ("R012", {"actual": "24312000000000001002", "expected": "无连号"}),
    ],
)
def test_任何建议都不教人规避审核(rule_id: str, ev: dict) -> None:
    (item,) = suggest_advice([_f(rule_id, **ev)])
    for banned in ("拆单", "拆分", "拆成", "绕过", "规避审核"):
        assert banned not in item["text"], f"{rule_id} 的建议出现了「{banned}」，这是在教人绕内控"


def test_只读_不改变输入也不产生通过项建议() -> None:
    passed = AuditFinding(rule_id="R001", clause="3.1", title="抬头", severity=Severity.PASS,
                          message="ok", evidence={})
    fail = _f("R010", claimed_amount=500.0, invoice_total=550.0)
    items = suggest_advice([passed, fail])
    assert len(items) == 1 and items[0]["rule_id"] == "R010"
    # 无副作用：函数返回新列表，原 findings 的严重度没被改动
    assert fail.severity is Severity.FAIL


# --------------------------------------------------------------------------
# 处置选项（suggest_dispositions）
# --------------------------------------------------------------------------


def _w(rule_id: str, **ev) -> AuditFinding:
    return AuditFinding(
        rule_id=rule_id, clause="4.2", title="测试用规则",
        severity=Severity.WARN, message="测试消息", evidence=ev,
    )


def test_限额超标给四条处置路径且模板留白() -> None:
    f = _f("R007", actual=800.0, expected="<= 600.00", nights=3, invoice_total=2400.0)
    opts = suggest_dispositions(
        [f], applicant="张三", expense_type="住宿费", issue_date="2026-09-15"
    )
    assert [o["code"] for o in opts] == ["revise", "supplement", "explain", "escalate"]
    explain = next(o for o in opts if o["code"] == "explain")
    assert "张三" in explain["detail"] and "住宿费" in explain["detail"]
    # 原因必须留白由本人填 —— 系统不替人编事实
    assert "____" in explain["detail"]
    assert explain["requires"] == "申请人 + 审批人"


def test_只命中疑点时不出现特批选项() -> None:
    opts = suggest_dispositions([_w("R008", actual=2690.0, expected="<= 2,000.00")])
    codes = [o["code"] for o in opts]
    assert "escalate" not in codes  # 没有事实性驳回，不该建议特批
    assert "supplement" in codes


def test_全部通过时没有处置选项() -> None:
    passed = AuditFinding(rule_id="R001", clause="3.1", title="抬头", severity=Severity.PASS,
                          message="ok", evidence={})
    assert suggest_dispositions([passed]) == []


def test_处置选项也不教规避() -> None:
    f = _f("R007", actual=800.0, expected="<= 600.00", nights=3, invoice_total=2400.0)
    for o in suggest_dispositions([f], applicant="张三", expense_type="住宿费"):
        for banned in ("拆单", "拆分", "拆成", "绕过", "规避审核"):
            assert banned not in o["detail"], f"{o['code']} 在教人绕内控"
