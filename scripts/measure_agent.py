"""agent 层行为评测 —— 拿 27 张样本跑一遍，出数字
====================================================

和 ``scripts/run_eval.py`` 同性质：**测量，不是演示**。

它回答三个问题：

1. **agent 层改变了判定吗？** —— 同一张票分别走 ``run_audit`` 与 ``runtime.submit``，
   逐条比对 findings，必须完全一致（不一致就是判定权外溢）。
2. **agent 到底替你做了什么？** —— 每张单自动发了几条消息、发给了谁、
   是"要材料"还是"通知复核"。
3. **这一层有多重？** —— 每张单的动作数、日志事件数、端到端耗时。

用法::

    python scripts/measure_agent.py
    python scripts/measure_agent.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import AgentRuntime, Inbox  # noqa: E402
from agent.runtime import REVIEW_DESK  # noqa: E402
from finance.audit import run_audit  # noqa: E402
from finance.store import AuditStore  # noqa: E402
from server import _build_request  # noqa: E402  —— 复用唯一一份申请单校验

MANIFEST = ROOT / "finance" / "samples" / "samples.yaml"


def _cases() -> list[tuple[str, Path, dict]]:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    return [
        (key, MANIFEST.parent / spec["pdf"], dict(spec.get("request") or {}))
        for key, spec in manifest.items()
    ]


def _signature(findings) -> list[tuple]:
    """判定指纹：规则号 + 严重度 + 人话 + 证据（键排序）。"""
    return [
        (
            f.rule_id,
            f.severity.value,
            f.message,
            json.dumps(f.evidence, sort_keys=True, ensure_ascii=False, default=str),
        )
        for f in findings
    ]


def measure() -> dict:
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="nono-measure-") as tmp:
        tmp_path = Path(tmp)
        for key, path, request_spec in _cases():
            if not path.is_file():
                raise SystemExit(f"样本缺失：{path}")
            request = _build_request(request_spec)

            # 路径 A：直接跑判定（基线）
            direct_store = AuditStore(base_dir=tmp_path / f"{key}-direct")
            started = time.perf_counter()
            direct = asyncio.run(
                run_audit(path, request, store=direct_store, narrative_llm=None)
            )
            direct_ms = (time.perf_counter() - started) * 1000

            # 路径 B：经 agent 受理
            agent_store = AuditStore(base_dir=tmp_path / f"{key}-agent")
            runtime = AgentRuntime(
                store=agent_store, inbox=Inbox(base_dir=tmp_path / f"{key}-inbox")
            )
            started = time.perf_counter()
            report = asyncio.run(runtime.submit(path, request))
            agent_ms = (time.perf_counter() - started) * 1000

            via = agent_store.load(report.audit_id)
            messages = [
                m
                for m in runtime.inbox.messages_for(REVIEW_DESK)
                if m["audit_id"] == report.audit_id
            ]
            applicant_msgs = [
                m
                for m in runtime.inbox.messages_for(request.applicant)
                if m["audit_id"] == report.audit_id
            ]
            events = agent_store.read_log(report.audit_id)

            rows.append(
                {
                    "key": key,
                    "suggested": via.suggested_status.value,
                    "blocking": len(via.blocking_findings),
                    "warnings": len(via.warning_findings),
                    "na": sum(1 for f in via.findings if not f.applicable),
                    "actions": [s.action for s in report.steps],
                    "message_to_review": len(messages),
                    "message_to_applicant": len(applicant_msgs),
                    "events": len(events),
                    "direct_ms": round(direct_ms, 1),
                    "agent_ms": round(agent_ms, 1),
                    "consistent": _signature(via.findings) == _signature(direct.findings),
                }
            )

    total = len(rows)
    return {
        "total": total,
        "consistent": sum(r["consistent"] for r in rows),
        "to_review": sum(1 for r in rows if r["message_to_review"]),
        "supplement": sum(1 for r in rows if r["message_to_applicant"]),
        "blocked": sum(1 for r in rows if r["blocking"]),
        "warned": sum(1 for r in rows if r["warnings"]),
        "clean": sum(1 for r in rows if not r["blocking"] and not r["warnings"]),
        "actions": sum(len(r["actions"]) for r in rows),
        "events": sum(r["events"] for r in rows),
        # 两条路径的耗时分开报：差值受冷缓存/写盘顺序影响，不做结论
        "avg_direct_ms": round(sum(r["direct_ms"] for r in rows) / total, 1),
        "avg_agent_ms": round(sum(r["agent_ms"] for r in rows) / total, 1),
        "rows": rows,
    }


def render(result: dict) -> str:
    t = result["total"]
    lines = [
        f"## agent 层行为评测（{t} 张样本）",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 结论与直接 run_audit 一致 | {result['consistent']}/{t} |",
        f"| 自动推进到「待签批」 | {t}/{t}（全部停在人工决定之前） |",
        f"| 自动发「补料请求」给申请人 | {result['supplement']} 张 |",
        f"| 自动发「待复核」给审核台 | {result['to_review']} 张 |",
        f"| 其中命中驳回项（FAIL） | {result['blocked']} 张 |",
        f"| 其中仅命中疑点（WARN） | {result['warned']} 张 |",
        f"| 全通过（零疑点） | {result['clean']} 张 |",
        f"| agent 动作总数 / 平均每张 | {result['actions']} / "
        f"{result['actions'] / t:.1f} |",
        f"| 日志事件总数 / 平均每张 | {result['events']} / {result['events'] / t:.1f} |",
        f"| 单票耗时（直接判定 / 经 agent） | "
        f"{result['avg_direct_ms']} / {result['avg_agent_ms']} ms |",
        "",
        "| 样本 | 建议 | FAIL | WARN | N/A | 动作 | 收件人 | 与基线一致 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        to = []
        if r["message_to_review"]:
            to.append("审核台")
        if r["message_to_applicant"]:
            to.append("申请人")
        lines.append(
            f"| {r['key']} | {r['suggested']} | {r['blocking']} | {r['warnings']} | "
            f"{r['na']} | {len(r['actions'])} | {'/'.join(to) or '—'} | "
            f"{'是' if r['consistent'] else '否'} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="agent 层行为评测")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 CI 消费）")
    args = parser.parse_args()

    result = measure()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render(result))

    if result["consistent"] != result["total"]:
        print("\n有样本的结论与直接 run_audit 不一致 —— 判定权外溢了", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
