"""评测脚本自身的测试
======================

评测是这个项目"规则真的在工作"的**唯一证据**。所以评测脚本自己也得被盯着 ——
一个只会打印 ✅ 的评测脚本比没有评测更糟：它给的是虚假的信心。

这里钉三件事（都是实测踩过的）：

1. ``expects`` 的解析 —— 规则号后面的严重度必须被读出来，且不能串行。
2. **严重度不符要算失败**。过去只查"命中没命中"，把 R007 从 FAIL 降级成
   WARN 照样算命中 —— 而这两个严重度一个走向驳回、一个走向转人工。
3. **误报要进失败列表与退出码**。过去 S01 误报 2 条时脚本照样打印
   「全部样本符合预期 ✅」并返回 0。

全程离线，不跑真样本（那部分由 run_eval.py 自己跑）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_eval  # noqa: E402


def make_outcome(**over) -> run_eval.SampleOutcome:
    base = dict(
        key="S99",
        title="测试样本",
        expected={"R007": "FAIL"},
        actual_non_pass={"R007": "FAIL"},
        suggested="REJECTED",
        extraction_ok=True,
    )
    base.update(over)
    return run_eval.SampleOutcome(**base)


# ---------------------------------------------------------------------------
# expects 解析
# ---------------------------------------------------------------------------


def test_parse_expects_reads_severity():
    assert run_eval.parse_expects("R007 FAIL（一线城市限额 600/晚）") == {"R007": "FAIL"}
    assert run_eval.parse_expects("R012 WARN（疑似拆单）") == {"R012": "WARN"}
    assert run_eval.parse_expects("R008 WARN（提交人工复核，不直接驳回）") == {"R008": "WARN"}


def test_parse_expects_handles_multiple_rules_in_one_sentence():
    parsed = run_eval.parse_expects("R001 FAIL（抬头不符）、R002 FAIL（税号为空）")
    assert parsed == {"R001": "FAIL", "R002": "FAIL"}


def test_parse_expects_does_not_attach_a_later_word_to_an_earlier_rule():
    """「R016 FAIL（大写 1560 ≠ 小写 1650）」里的别的词不能被粘成严重度。"""
    assert run_eval.parse_expects("R016 FAIL（大写 1560 ≠ 小写 1650）") == {"R016": "FAIL"}


def test_parse_expects_ignores_sentences_without_rule_ids():
    """S01 那句「全部 17 条 PASS」是**叙述**不是规则号，不能被当成期望。"""
    assert run_eval.parse_expects("全部 17 条 PASS，系统建议 APPROVED") == {}
    assert run_eval.parse_expects("全通过（批内第一张，无历史可比）") == {}


# ---------------------------------------------------------------------------
# 严重度
# ---------------------------------------------------------------------------


def test_severity_downgrade_is_a_failure():
    """命中但严重度不符 -> 失败。

    反证：把 R007 从 FAIL 降级为 WARN，规则确实"命中了"，
    但系统建议会从 REJECTED 变成 PENDING —— 结论完全不同。
    这种改动必须让评测红，否则评测只是在数"有没有报错"。
    """
    outcome = make_outcome(actual_non_pass={"R007": "WARN"})
    assert outcome.severity_mismatch == ["R007 期望 FAIL、实际 WARN"]
    assert not outcome.hits_expected
    assert outcome.is_bad


def test_severity_not_stated_means_hit_only():
    """``expects`` 没写严重度时，只查命中 —— 不替样本清单加戏。"""
    outcome = make_outcome(expected={"R007": None})
    assert outcome.severity_mismatch == []
    assert outcome.hits_expected


# ---------------------------------------------------------------------------
# 误报
# ---------------------------------------------------------------------------


def test_false_positive_makes_the_sample_fail():
    """误报必须计入失败。

    反证：S01 本该全通过，若某条规则误报，过去只挂 ⚠️ 不进 ``failed``，
    脚本照样返回 0 —— "只报喜不报忧"的评测等于没有评测。
    """
    outcome = make_outcome(expected={}, actual_non_pass={"R014": "WARN"})
    assert outcome.false_positives == ["R014"]
    assert outcome.hits_expected
    assert outcome.is_bad


def test_render_reports_false_positives_and_severity_mismatches():
    """误报与严重度不符都要出现在指标里，并让整轮评测判不通过。"""
    clean = make_outcome(key="S01", expected={}, actual_non_pass={}, suggested="APPROVED")
    # 单纯误报：该报的一条都没报错，但多报了一条
    noisy = make_outcome(
        key="S02", expected={}, actual_non_pass={"R014": "WARN"}, suggested="PENDING"
    )
    # 严重度不符：命中了，但状态机走向完全不同
    downgraded = make_outcome(
        key="S03",
        expected={"R007": "FAIL"},
        actual_non_pass={"R007": "WARN"},
        suggested="PENDING",
    )
    report, metrics = run_eval.render([clean, noisy, downgraded])

    assert metrics["false_positives"] == 1
    assert metrics["severity_mismatches"] == ["R007 期望 FAIL、实际 WARN"]
    assert metrics["failed"] == ["S02", "S03"]
    assert "有误报（计入失败）" in report
    assert "❌ 与预期不符" in report
    assert "严重度一致 | 1" in report


def test_render_is_clean_when_everything_matches():
    clean = make_outcome(expected={}, actual_non_pass={}, suggested="APPROVED")
    report, metrics = run_eval.render([clean])
    assert metrics["failed"] == []
    assert "✅" in report


# ---------------------------------------------------------------------------
# 抽取核对
# ---------------------------------------------------------------------------


def test_check_extraction_has_no_skip_heuristic():
    """抬头不是公司全称时，**不能**整张票跳过核对。

    回归测试：旧代码写着「buyer_name 非空且不等于公司全称 -> return True」，
    本意是放过 S04（抬头本来就是"个人"）。可它是个通用开关 ——
    任何一张票抬头被抽错，整张票的字段核对全部跳过，指标照样满分。
    """
    from finance.models import Invoice

    spec = {
        "invoice": {
            "buyer_name": "个人",
            "invoice_number": "24312000000012345604",
            "total": "600.00",
        }
    }
    ok = Invoice(
        invoice_number="24312000000012345604", buyer_name="个人", total="600.00",
        invoice_type="电子发票（普通发票）",
    )
    assert run_eval.check_extraction(ok, spec) == (True, [])

    # 抬头抽错 -> 必须报出来，而不是"跳过核对"
    wrong = Invoice(
        invoice_number="24312000000012345604", buyer_name="某公司", total="600.00",
        invoice_type="电子发票（普通发票）",
    )
    passed, notes = run_eval.check_extraction(wrong, spec)
    assert not passed and any("购买方名称" in n for n in notes)


def test_check_extraction_reports_missing_invoice_section():
    """清单里没有 invoice 段就是失败，不能默认通过。"""
    from finance.models import Invoice

    passed, notes = run_eval.check_extraction(Invoice(), {})
    assert not passed
    assert "invoice" in notes[0]


@pytest.mark.parametrize(
    "field, value, expect_note",
    [
        ("invoice_type", "", "发票类型"),
        ("invoice_number", "", "发票号码"),
        ("total", None, "价税合计"),
    ],
)
def test_check_extraction_catches_empty_fields(field, value, expect_note):
    from finance.models import Invoice

    spec = {"invoice": {field: "样本定义值"}}
    invoice = Invoice(**{field: value})
    passed, notes = run_eval.check_extraction(invoice, spec)
    assert not passed and any(expect_note in n for n in notes)
