"""推翻统计 —— 把"人不同意系统"变成可审计的报告
==================================================

这个脚本回答一个问题：**哪些规则最常被人推翻？**

为什么值得做：系统建议 `REJECTED` 而人工批了，或反过来，都是**制度与现实的
摩擦点**。摩擦点集中在哪几条规则上，就是制度该修订的地方。

它**不是** RLHF
--------------
这里没有任何"自动优化"：规则不自动改、阈值不自动调、prompt 不自动改。
报告只输出给人看，改不改由人决定 —— 改规则要走制度修订 + 代码评审 + 版本号。
理由：判定权一旦跟着历史决定漂移，这个系统就不再可复算了。

数据从哪来
----------
``decided`` 事件里 ``overridden=true`` 的记录 —— 人在推翻系统建议时**被强制**
填写的书面理由（制度 2.3，服务端 400 拦截）。这批数据是真实使用后才会积累的，
27 张合成样本跑不出一条（没人做决定）。

归因口径
--------
- 建议 `REJECTED` → 人批了：归因到**命中驳回项的那些规则**（FAIL）；
- 建议 `APPROVED` → 人驳了：**没有规则可归因**，计入"人工自由裁量"——
  系统当时没意见，人不同意，这不是某条规则的错。
- 建议 `PENDING` 时人做决定不算"推翻"（系统拒绝表态，不是有意见），
  但也会出现在"已决定"分母里。

用法::

    python scripts/measure_overrides.py
    python scripts/measure_overrides.py --month 2026-10
    python scripts/measure_overrides.py --data-dir D:/elsewhere/data/audits --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance.store import AuditStore  # noqa: E402


def collect(store: AuditStore) -> list[dict[str, Any]]:
    """读全部已决定的单据，返回逐条记录（含归因用的驳回项）。"""
    rows: list[dict[str, Any]] = []
    for result in store.all_records():
        if result.decision is None:
            continue
        events = store.read_log(result.audit_id)
        decided = [e for e in events if e.get("event") == "decided"]
        if not decided:
            continue
        event = decided[-1]

        when = result.decided_at
        month = when.strftime("%Y-%m") if when else str(event.get("ts", ""))[:7]
        rows.append(
            {
                "audit_id": result.audit_id,
                "month": month,
                "operator": str(event.get("operator") or result.operator or ""),
                "suggested": str(event.get("suggested_status") or ""),
                "decision": str(event.get("decision") or ""),
                "overridden": bool(event.get("overridden")),
                "reason": str(event.get("override_reason") or ""),
                "rules": [
                    {"rule_id": f.rule_id, "clause": f.clause}
                    for f in result.blocking_findings
                ],
                "warnings": [f.rule_id for f in result.warning_findings],
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overridden = [r for r in rows if r["overridden"]]
    by_rule: Counter[str] = Counter()
    clauses: dict[str, str] = {}
    for row in overridden:
        for finding in row["rules"]:
            by_rule[finding["rule_id"]] += 1
            clauses.setdefault(finding["rule_id"], finding["clause"])

    return {
        "decided": len(rows),
        "overridden": len(overridden),
        "rate": round(len(overridden) / len(rows), 3) if rows else 0.0,
        "by_rule": dict(by_rule.most_common()),
        "clauses": clauses,
        "by_operator": dict(Counter(r["operator"] or "（未署名）" for r in overridden).most_common()),
        "by_month": dict(sorted(Counter(r["month"] for r in overridden).items())),
        "decided_by_month": dict(sorted(Counter(r["month"] for r in rows).items())),
        "unattributed": sum(1 for r in overridden if not r["rules"]),
        "reasons": [r["reason"] for r in overridden if r["reason"]][:10],
        "rows": rows,
    }


def render(result: dict[str, Any]) -> str:
    if result["decided"] == 0:
        return (
            "暂无已做决定的单据。\n"
            "推翻统计依赖真实使用：`decided` 事件里 `overridden=true` 的记录"
            "只有在人推翻系统建议时才会产生。"
        )

    lines = [
        "## 人工推翻统计",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 已决定 | {result['decided']} |",
        f"| 推翻系统建议 | {result['overridden']}（{result['rate']:.1%}） |",
        f"| 其中无规则可归因（人工自由裁量） | {result['unattributed']} |",
    ]

    if result["by_rule"]:
        lines += ["", "### 被推翻最多的规则", "", "| 规则 | 条款 | 次数 |", "|---|---|---|"]
        for rule_id, count in result["by_rule"].items():
            lines.append(f"| {rule_id} | {result['clauses'].get(rule_id, '—')} | {count} |")

    if result["by_operator"]:
        lines += ["", "### 按操作人", "", "| 操作人 | 推翻次数 |", "|---|---|"]
        for name, count in result["by_operator"].items():
            lines.append(f"| {name} | {count} |")

    if result["by_month"]:
        lines += ["", "### 按月", "", "| 月份 | 推翻 / 已决定 |", "|---|---|"]
        for month, count in result["by_month"].items():
            lines.append(f"| {month} | {count} / {result['decided_by_month'].get(month, 0)} |")

    if result["reasons"]:
        lines += ["", "### 理由样本（最近 10 条）", ""]
        for reason in result["reasons"]:
            lines.append(f"- {reason}")

    lines += [
        "",
        "> 规则不会根据这份统计自动调整。它输出给人看："
        "某条规则被反复推翻，说明制度条款或阈值与现实有摩擦，"
        "该改制度（走修订 + 代码评审 + 升版本号），不该让系统跟着历史决定漂移。",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="人工推翻统计（可审计版，不是 RLHF）")
    parser.add_argument("--data-dir", default="", help="审计单目录（默认 data/audits）")
    parser.add_argument("--month", default="", help="只看某个月，如 2026-10")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 CI/看板消费）")
    args = parser.parse_args()

    store = AuditStore(base_dir=args.data_dir) if args.data_dir else AuditStore()
    rows = collect(store)
    if args.month:
        rows = [r for r in rows if r["month"] == args.month]
    result = summarize(rows)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
