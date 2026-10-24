"""演示：一张票进来，agent 自己跑到"待签批"
==========================================

不启动 Web 服务，直接在命令行看完整链路：

    抽取 → 19 条规则 → 凭证草稿 → agent 分派动作 → 落审计日志

跑完你会看到三样东西：
1. 报告：这一步是谁触发的、做了什么；
2. 收件箱：agent 发给谁的消息（站内，不发真邮件）；
3. 时间线：每一步的 actor_type（system / agent / human）。

注意它**到"待签批"就停了** —— 后面的批准、付款、归档，
agent 没有那把钥匙（见 agent/gate.py）。

用法::

    python scripts/agent_demo.py
    python scripts/agent_demo.py --sample S02_hotel_over_limit.pdf --amount 2400.00
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import AgentRuntime, Inbox  # noqa: E402  —— 先补 sys.path 再导入
from agent.runtime import REVIEW_DESK  # noqa: E402
from finance.models import ReimbursementRequest  # noqa: E402
from finance.store import AuditStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="受控 agent 单票演示")
    ap.add_argument("--sample", default="S01_hotel_ok.pdf", help="finance/samples/pdf 下的文件名")
    ap.add_argument("--amount", default="1650.00", help="申请金额")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "demo"), help="演示数据目录")
    ap.add_argument("--recover", action="store_true", help="顺便演示断点恢复")
    args = ap.parse_args()

    pdf = ROOT / "finance" / "samples" / "pdf" / args.sample
    if not pdf.is_file():
        print(f"找不到样本票：{pdf}")
        return 1

    base = Path(args.data_dir)
    store = AuditStore(base_dir=base / "audits")
    inbox = Inbox(base_dir=base / "inbox")
    runtime = AgentRuntime(store=store, inbox=inbox)

    request = ReimbursementRequest(
        applicant="张三",
        department="技术部",
        expense_type="住宿费",
        amount=args.amount,
        reason="上海客户现场支持",
        submit_date=date.today(),
        city="上海",
        nights=3,
    )

    report = asyncio.run(runtime.submit(pdf, request))

    print(f"\n单号 {report.audit_id}　状态 {report.state}　系统建议 {report.suggested_status}")
    print("agent 做的动作：")
    for step in report.steps:
        print(f"  - {step.action} → {step.outcome}（收件人：{step.to}）")

    print("\n收件箱：")
    for who in {REVIEW_DESK, request.applicant}:
        for msg in runtime.inbox.messages_for(who):
            if msg["audit_id"] == report.audit_id:
                print(f"  [{who}] {msg['subject']}")

    print("\n时间线（谁做了什么）：")
    for row in runtime.timeline(report.audit_id):
        label = row.get("stage") or row.get("action") or row.get("event")
        print(f"  seq={row['seq']:<3} {row['actor_type']:<7} {label}")

    if args.recover:
        print("\n断点恢复（只补动作，不重跑判定）：")
        for rec in runtime.recover():
            print(f"  {rec.audit_id}: {[s.outcome for s in rec.steps]}")

    print(f"\n数据目录：{base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
