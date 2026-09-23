"""复检回归（Claude / Codex / Trae 三家实测点名的残留，逐条销项）
================================================================

这批测试的存在理由很直白：三家评测都在干同一件事 —— 拿实测把"声称修好了"
逐条兑现。这里把它们报过的每个残留各钉一颗钉子，改回去就红。
"""

from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

import pytest

from finance import Severity, evaluate, load_policy_bundle, parse_money
from finance.models import Invoice, ReimbursementRequest
from finance.policy import PolicyError, parse_clause_headings
from finance.store import MemoryHistoryView
from finance.rules import CHECKERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def policy():
    return load_policy_bundle()


def _invoice(**over) -> Invoice:
    base = dict(
        invoice_number="24312000000000000001",
        invoice_type="电子发票（普通发票）",
        issue_date=date(2026, 9, 15),
        buyer_name="示例科技有限公司",
        buyer_tax_id="91310000MA1FL2XXXX",
        seller_name="上海某某酒店管理有限公司",
        item_name="*住宿服务*住宿费",
        amount="1556.60",
        tax_rate="6%",
        tax_amount="93.40",
        total="1650.00",
        total_in_words="壹仟陆佰伍拾圆整",
    )
    base.update(over)
    return Invoice(**base)


def _request(**over) -> ReimbursementRequest:
    base = dict(
        applicant="张三", department="技术部", expense_type="住宿费",
        amount="1650.00", reason="客户支持", submit_date=date(2026, 9, 18),
    )
    base.update(over)
    return ReimbursementRequest(**base)


def _finding(findings, rule_id):
    for f in findings:
        if f.rule_id == rule_id:
            return f
    raise AssertionError(f"没有跑出规则 {rule_id}")


# ---------------------------------------------------------------------------
# C6：_skip 路径必须打 N/A —— "没查"不能冒充"查了没事"
# ---------------------------------------------------------------------------


def test_c6_skip_marks_not_applicable(policy):
    inv = _invoice(item_name="*运输服务*市内交通", tax_rate="9%", amount=None,
                   tax_amount=None, total="380.00", total_in_words="叁佰捌拾圆整")
    findings = evaluate(inv, _request(expense_type="市内交通费", amount="380.00",
                                      nights=None), policy,
                        history=MemoryHistoryView())
    for rid in ("R006", "R007", "R008"):
        f = _finding(findings, rid)
        assert f.applicable is False, f"{rid} 费用类型不涉及 -> 必须 N/A"
        assert "不适用" in f.message or "不涉及" in f.message or "非" in f.message


# ---------------------------------------------------------------------------
# C4 残留：配置缺一半就炸；两块全缺 = 不路由（全规则照跑）
# ---------------------------------------------------------------------------


def test_c4_partial_config_is_a_policy_error():
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(suffix="pol")) / "policies"
    shutil.copytree(PROJECT_ROOT / "finance" / "policies", tmp)
    p = tmp / "rules.yaml"
    text = p.read_text(encoding="utf-8")
    text = text.replace("rule_scope:", "rule_scope_disabled:")
    p.write_text(text, encoding="utf-8")
    with pytest.raises(PolicyError):
        load_policy_bundle(tmp)


def test_c4_missing_config_disables_routing(policy):
    # 两块全缺 -> ticket_kind 返回 unknown -> 不跳过任何规则（宁多勿漏）
    clone = policy
    saved = (clone.ticket_kinds, clone.rule_scope)
    clone.ticket_kinds, clone.rule_scope = {}, {}
    try:
        assert clone.ticket_kind("电子发票（普通发票）") == "unknown"
        assert clone.rule_in_scope("R001", "unknown") is True
    finally:
        clone.ticket_kinds, clone.rule_scope = saved


# ---------------------------------------------------------------------------
# C1：定额发票 = 制度 3.4 明文不接受 -> FAIL（曾被"按票种核验"全绿放行）
# ---------------------------------------------------------------------------


def test_c1_fixed_voucher_is_rejected_by_policy(policy):
    inv = _invoice(invoice_type="定额发票", item_name="*运输服务*出租车费",
                   amount=None, tax_amount=None, tax_rate="",
                   total="100.00", total_in_words="壹佰圆整", buyer_tax_id="")
    findings = evaluate(inv, _request(expense_type="市内交通费", amount="100.00"),
                        policy, history=MemoryHistoryView())
    f = _finding(findings, "R009")
    assert f.severity is Severity.FAIL
    assert "不得报销" in f.message


# ---------------------------------------------------------------------------
# INVOICE（裸英文票名）-> foreign，R019 才触发
# ---------------------------------------------------------------------------


def test_bare_english_invoice_is_foreign(policy):
    assert policy.ticket_kind("INVOICE") == "foreign"
    assert policy.ticket_kind("境外 Invoice") == "foreign"


# ---------------------------------------------------------------------------
# B2 残留：证据链元数据不接受申报口覆盖
# ---------------------------------------------------------------------------


def test_b2_protected_metadata_cannot_be_overridden():
    from fastapi.testclient import TestClient

    import server

    client = TestClient(server.app)
    resp = client.post(
        "/api/audit/run",
        json={
            "filename": "x.pdf",
            "content_b64": base64.b64encode(b"%PDF-1.4\n%%EOF").decode(),
            "request": {
                "applicant": "张三", "department": "技术部",
                "expense_type": "住宿费", "amount": "1650.00",
                "reason": "x", "submit_date": "2026-09-18",
            },
            "invoice_overrides": {"extraction_method": "pdf_text",
                                  "raw_text": "/etc/passwd"},
            "critical_confirmed": True,
        },
    )
    assert resp.status_code == 400
    assert "证据链元数据" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 文档-代码一致性：README 不许再说谎（三家连抓三轮的计数漂移，从此测试挂红）
# ---------------------------------------------------------------------------


def test_readme_claims_match_code(policy):
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    # ① 规则条数与真实规则表一致，旧数字必须不存在
    assert f"{len(policy.rules)} 条规则" in readme
    assert "18 条规则" not in readme
    # ② 规则表含全部规则（曾漏 R019）
    for rid in policy.rule_ids():
        assert rid in readme, f"README 规则表漏了 {rid}"
    # ③ 不许写死用例数（"155 passed"式数字必漂移，已被抓三轮）
    import re

    assert not re.search(r"\d+\s+passed", readme), "README 不许写死 passed 数字"
    # ④ 旧账清零：六态实为八态、money_eq 不再被引为合规依据
    assert "六态" not in readme
    assert "money_eq" not in readme


def test_clause_parser_covers_bold_number_only_headings(policy):
    """条款解析器两种写法都得认 —— 2.3（推翻必须写理由）曾整条隐形。"""
    headings = parse_clause_headings(policy.policy_md)
    assert "2.3" in headings
    assert "3.1" in headings


def test_state_machine_wording_is_eight():
    audit_src = (PROJECT_ROOT / "finance" / "audit.py").read_text(encoding="utf-8")
    models_src = (PROJECT_ROOT / "finance" / "models.py").read_text(encoding="utf-8")
    # 只禁"六态状态机"这个旧说法 —— "第五态到第六态"这类正常措辞不许误伤
    assert "六态状态机" not in audit_src and "八态" in audit_src
    assert "六态" not in models_src


def test_every_checker_registered(policy):
    for spec in policy.rules:
        assert spec.checker in CHECKERS
