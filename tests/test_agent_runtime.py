"""agent 层的行为与边界
======================

测的是"它能做对事"，更是"它做不了不该做的事"。

和 test_finance.py 的四条不变量同一套思路：把**主张**写成断言，
以后谁把能力放大，先红一遍再说。
"""

from __future__ import annotations

import asyncio
import ast
import json
from datetime import date
from pathlib import Path

import pytest

from agent import (
    AGENT_TOOL_ACTIONS,
    Action,
    Gate,
    Inbox,
    OutOfBoundsAction,
    agent_actor,
    human,
)
from agent.runtime import REVIEW_DESK, AgentRuntime
from finance.audit import decide, run_audit
from finance.models import (
    AuditState,
    Decision,
    ReimbursementRequest,
    SuggestedStatus,
)
from finance.store import AuditStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PDF = PROJECT_ROOT / "finance" / "samples" / "pdf"


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------


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


@pytest.fixture
def store(tmp_path) -> AuditStore:
    return AuditStore(base_dir=tmp_path / "audits")


@pytest.fixture
def runtime(store, tmp_path) -> AgentRuntime:
    return AgentRuntime(store=store, inbox=Inbox(base_dir=tmp_path / "inbox"))


def _submit(runtime, pdf_name: str, **req_over):
    pdf = SAMPLES_PDF / pdf_name
    if not pdf.is_file():
        pytest.skip(f"样本票 {pdf_name} 缺失，请先运行 scripts/make_samples.py")
    return asyncio.run(runtime.submit(pdf, make_request(**req_over)))


def _events(store, audit_id):
    return store.read_log(audit_id)


def _strip_finished(store, audit_id):
    """模拟"写完 started / action 但进程在 finished 之前挂了"。

    只删**尾部**那条 finished，哈希链仍然完整 —— 恢复逻辑不该依赖"日志坏了"。
    """
    path = store.base_dir / f"{audit_id}.log.jsonl"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    while lines:
        tail = json.loads(lines[-1])
        if tail.get("event") == "agent.run.finished":
            lines.pop()
            continue
        break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# 1. 全通过：自动通知审核台，停在待签批
# --------------------------------------------------------------------------


def test_submit_dispatches_draft_ready_and_stops_before_signoff(runtime, store):
    report = _submit(runtime, "S01_hotel_ok.pdf")

    assert report.actions == ["notify_reviewer"]
    assert report.state == AuditState.PENDING_REVIEW.value

    saved = store.load(report.audit_id)
    # agent 没有、也不能替人决定
    assert saved.decision is None
    assert saved.state is AuditState.PENDING_REVIEW
    assert saved.suggested_status is SuggestedStatus.APPROVED

    messages = runtime.inbox.messages_for(REVIEW_DESK)
    assert len(messages) == 1
    assert messages[0]["kind"] == "draft_ready"
    assert messages[0]["audit_id"] == report.audit_id
    assert "草稿不是凭证" in messages[0]["body"]


def test_agent_events_carry_actor_and_are_hash_chained(runtime, store):
    report = _submit(runtime, "S01_hotel_ok.pdf")
    events = _events(store, report.audit_id)

    started = [e for e in events if e["event"] == "agent.run.started"]
    assert len(started) == 1
    assert started[0]["actor"] == "nono-agent"
    assert started[0]["actor_type"] == "agent"
    assert started[0]["input"] == "S01_hotel_ok.pdf"
    assert len(started[0]["input_sha256"]) == 64
    # 没调模型就必须如实写 template，不能假装用过 AI
    assert started[0]["narrative_source"] == "template"

    actions = [e for e in events if e["event"] == "agent.action"]
    assert actions and actions[0]["actor_type"] == "agent"
    assert actions[0]["result"] == "sent"

    assert store.verify_log(report.audit_id)["ok"] is True


# --------------------------------------------------------------------------
# 2. 驳回项：通知审核台，不自行驳回
# --------------------------------------------------------------------------


def test_blocking_findings_notify_reviewer_not_reject(runtime, store):
    report = _submit(runtime, "S02_hotel_over_limit.pdf", amount="2400.00")

    saved = store.load(report.audit_id)
    assert saved.suggested_status is SuggestedStatus.REJECTED
    assert saved.decision is None  # 系统只给建议

    messages = runtime.inbox.messages_for(REVIEW_DESK)
    assert len(messages) == 1
    assert messages[0]["kind"] == "review_required"
    assert "命中驳回项" in messages[0]["body"]
    assert "系统不会自行批准或驳回" in messages[0]["body"]


# --------------------------------------------------------------------------
# 3. 疑点缺料：向申请人要材料（L1），不是向审核人要判断
# --------------------------------------------------------------------------


def test_warning_findings_request_info_from_applicant(runtime):
    report = _submit(
        runtime,
        "S08_meal_no_headcount.pdf",
        expense_type="餐饮费",
        amount="600.00",
        nights=None,
        headcount=None,
        reason="客户工作餐",
    )

    assert report.actions == ["request_info"]
    messages = runtime.inbox.messages_for("张三")
    assert len(messages) == 1
    assert messages[0]["kind"] == "supplement"
    assert messages[0]["to"] == "张三"
    assert "补齐后重新提交" in messages[0]["body"]
    # 给审核台不该收到"要材料"的消息 —— 收件人错位就是流程错位
    assert runtime.inbox.messages_for(REVIEW_DESK) == []


# --------------------------------------------------------------------------
# 4. 闸门：L3 动作 agent 根本没有
# --------------------------------------------------------------------------


def test_agent_cannot_reach_l3_actions():
    gate = Gate()
    for action in (Action.APPROVE, Action.PAY, Action.FILE_ARCHIVE):
        with pytest.raises(OutOfBoundsAction) as exc:
            gate.check(action, agent_actor())
        assert "L3" in str(exc.value)
        assert exc.value.log_fields()["legal_basis"]  # 拒绝必须带法律依据

    # 工具注册源里没有 L3 —— 不是"拦住"，是"不给"
    assert Action.PAY not in AGENT_TOOL_ACTIONS
    assert Action.APPROVE not in AGENT_TOOL_ACTIONS
    assert Action.FILE_ARCHIVE not in AGENT_TOOL_ACTIONS
    assert Action.NOTIFY_REVIEWER in AGENT_TOOL_ACTIONS


def test_human_may_do_l3_by_gate():
    gate = Gate()
    gate.check(Action.PAY, human("李四", role="出纳"))  # 不抛
    gate.check(Action.FILE_ARCHIVE, human("王五", role="会计主管"))


def test_out_of_bounds_is_logged_when_dispatch_tries_it(runtime, store):
    """越级尝试要留证据：日志里有 agent.gate.denied，而不是悄悄失败。"""
    report = _submit(runtime, "S01_hotel_ok.pdf")
    saved = store.load(report.audit_id)

    step = runtime._act(
        Action.PAY,
        runtime.agent,
        saved,
        to="出纳",
        kind="reminder",
        dedupe_key="x",
        subject="s",
        body="b",
    )
    assert step.outcome == "denied"
    denied = [e for e in _events(store, report.audit_id) if e["event"] == "agent.gate.denied"]
    assert len(denied) == 1
    assert denied[0]["action"] == "pay"
    assert store.verify_log(report.audit_id)["ok"] is True


# --------------------------------------------------------------------------
# 5. 恢复：只补动作，不重跑判定（这是审计上最重要的一条）
# --------------------------------------------------------------------------


def test_recover_replays_actions_without_reauditing(runtime, store):
    report = _submit(runtime, "S01_hotel_ok.pdf")
    before = _events(store, report.audit_id)
    stage_before = [e for e in before if e.get("event") == "stage"]

    _strip_finished(store, report.audit_id)
    assert not [
        e for e in _events(store, report.audit_id) if e.get("event") == "agent.run.finished"
    ]

    reports = runtime.recover()
    assert len(reports) == 1 and reports[0].recovered is True

    after = _events(store, report.audit_id)
    stage_after = [e for e in after if e.get("event") == "stage"]
    # 判定没有被重跑：阶段事件一条不多一条不少
    assert len(stage_after) == len(stage_before)
    # 动作没有重复投递（幂等）
    assert len(runtime.inbox.messages_for(REVIEW_DESK)) == 1
    assert reports[0].steps[0].outcome == "deduped"
    assert store.verify_log(report.audit_id)["ok"] is True


def test_recover_skips_decided_and_unfinished(runtime, store):
    report = _submit(runtime, "S01_hotel_ok.pdf")
    saved = store.load(report.audit_id)
    decide(saved, Decision.APPROVED, "李会计", store=store)

    assert runtime.recover() == []


def test_recover_leaves_midway_states_to_humans(runtime, store):
    """停在 pending_review 之前（判定没跑完）的单子，恢复逻辑不碰。"""
    pdf = SAMPLES_PDF / "S01_hotel_ok.pdf"
    if not pdf.is_file():
        pytest.skip("样本票缺失")
    result = asyncio.run(run_audit(pdf, make_request(), store=store, narrative_llm=None))
    result.state = AuditState.VALIDATED
    store.save(result)

    assert runtime.recover() == []


# --------------------------------------------------------------------------
# 6. 时间线：谁做了什么，能一眼看出来
# --------------------------------------------------------------------------


def test_timeline_labels_actor_type(runtime, store):
    report = _submit(runtime, "S01_hotel_ok.pdf")
    saved = store.load(report.audit_id)
    decide(saved, Decision.APPROVED, "李会计", store=store)

    line = runtime.timeline(report.audit_id)
    agent_rows = [r for r in line if r["event"].startswith("agent.")]
    human_rows = [r for r in line if r["event"] == "decided"]
    system_rows = [r for r in line if r["event"] == "stage"]

    assert agent_rows and all(r["actor_type"] == "agent" for r in agent_rows)
    assert human_rows and human_rows[0]["actor_type"] == "human"
    assert system_rows and all(r["actor_type"] == "system" for r in system_rows)
    # seq 连续，UI 可以直接按它排
    seqs = [r["seq"] for r in line]
    assert seqs == sorted(seqs)


# --------------------------------------------------------------------------
# 7. 结构性：agent 层不能绕过 finance 改状态
# --------------------------------------------------------------------------


def test_agent_layer_cannot_mutate_audit_state():
    """agent 包里的代码不许自己改状态、不许自己保存、不许自己决定。

    这就是"外挂"这个词的实际含义：它只能调 ``run_audit`` 和 append_log，
    别的门它连钥匙孔都没有。

    用 AST 而不是正则扫源码：docstring 里"提到"这些名字是正常的（本文档就提了），
    真正要禁的是**语法树里真的出现赋值/调用**。
    """
    agent_dir = PROJECT_ROOT / "agent"
    offenders: list[str] = []
    for py in sorted(agent_dir.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    _flag_mutation(node, target, py.name, offenders)
            elif isinstance(node, ast.AnnAssign):
                _flag_mutation(node, node.target, py.name, offenders)
            elif isinstance(node, ast.Call):
                name = _call_name(node)
                if name in {"decide", "save"}:
                    offenders.append(f"{py.name}:{node.lineno} 调用了 {name}()")
    assert not offenders, f"agent 层越界：{offenders}"


def _flag_mutation(node, target, filename: str, offenders: list[str]) -> None:
    if isinstance(target, ast.Attribute) and target.attr in {"state", "decision"}:
        offenders.append(f"{filename}:{node.lineno} 直接改了 .{target.attr}")


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""
