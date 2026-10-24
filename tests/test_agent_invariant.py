"""不变量：agent 层不改变任何判定
================================

这条测试把"只是完善，不是改变"钉成可执行的东西：

**同一张票走两条路径——直接 ``run_audit``、经 ``runtime.submit`` 受理——
逐条比对 findings、系统建议、凭证、状态、叙述，必须完全相等。**

允许 agent 做的只有"追加"：日志里多出 ``agent.*`` 事件与分派动作。
除此之外任何一处不同，都说明判定权被搬动过 —— 那是改变，不是完善。

一个实现细节：两边都先落盘再读回（走一遍 ``to_json_dict`` → ``model_validate``），
否则比较的是"Decimal 还是 float"这种序列化差异，而不是判定差异。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from agent import AgentRuntime, Inbox
from finance.audit import run_audit
from finance.store import AuditStore
from server import _build_request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "finance" / "samples" / "samples.yaml"


def _load_cases() -> list[tuple[str, Path, dict]]:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    return [
        (key, MANIFEST.parent / spec["pdf"], dict(spec.get("request") or {}))
        for key, spec in manifest.items()
    ]


CASES = _load_cases()


def _findings_signature(findings) -> list[tuple]:
    """结论的规范化指纹：规则号、严重度、人话、证据（键排序）。"""
    return [
        (
            f.rule_id,
            f.severity.value,
            f.message,
            f.applicable,
            json.dumps(f.evidence, sort_keys=True, ensure_ascii=False, default=str),
        )
        for f in findings
    ]


def _audit_events(store: AuditStore, audit_id: str) -> list[tuple]:
    """只看 audit 自己写的轨迹；agent.* 是新增，不参与"相等"的比较。"""
    return [
        (e.get("event"), e.get("stage"), e.get("result"))
        for e in store.read_log(audit_id)
        if not str(e.get("event", "")).startswith("agent.")
    ]


@pytest.mark.parametrize("key, path, request_spec", CASES, ids=[c[0] for c in CASES])
def test_agent_does_not_change_any_conclusion(key, path, request_spec, tmp_path):
    if not path.is_file():
        pytest.skip(f"样本缺失：{path.name}")
    request = _build_request(request_spec)

    direct_store = AuditStore(base_dir=tmp_path / "direct")
    direct = asyncio.run(run_audit(path, request, store=direct_store, narrative_llm=None))
    direct_store.save(direct)
    direct = direct_store.load(direct.audit_id)  # 同一套序列化往返

    agent_store = AuditStore(base_dir=tmp_path / "agent")
    runtime = AgentRuntime(store=agent_store, inbox=Inbox(base_dir=tmp_path / "inbox"))
    report = asyncio.run(runtime.submit(path, request))
    via = agent_store.load(report.audit_id)
    assert via is not None

    assert _findings_signature(via.findings) == _findings_signature(direct.findings), (
        f"{key}：agent 路径的判定结果与原路径不一致"
    )
    assert via.suggested_status is direct.suggested_status
    assert via.state is direct.state
    assert via.decision is None and direct.decision is None
    assert via.narrative == direct.narrative
    assert via.narrative_source is direct.narrative_source

    assert (via.voucher is None) == (direct.voucher is None)
    if direct.voucher is not None:
        assert via.voucher.balanced == direct.voucher.balanced
        assert [line.model_dump() for line in via.voucher.lines] == [
            line.model_dump() for line in direct.voucher.lines
        ]

    # 轨迹：audit 自己的事件逐条一致；agent 只做追加
    assert _audit_events(agent_store, via.audit_id) == _audit_events(
        direct_store, direct.audit_id
    )


def test_agent_events_are_the_only_addition(tmp_path):
    """反过来说：agent 路径**必须**多出事件，否则"完善"没有落到轨迹上。"""
    pdf = MANIFEST.parent / "pdf" / "S01_hotel_ok.pdf"
    if not pdf.is_file():
        pytest.skip("样本缺失")
    request = _build_request(dict(CASES[0][2]))

    agent_store = AuditStore(base_dir=tmp_path / "agent")
    runtime = AgentRuntime(store=agent_store, inbox=Inbox(base_dir=tmp_path / "inbox"))
    report = asyncio.run(runtime.submit(pdf, request))

    events = [e.get("event") for e in agent_store.read_log(report.audit_id)]
    agent_rows = [e for e in events if str(e).startswith("agent.")]
    assert agent_rows, "经 agent 受理却没有 agent.* 事件"
    # 判定阶段的记录一条不少
    assert events.index("submitted") < events.index("agent.run.started")
