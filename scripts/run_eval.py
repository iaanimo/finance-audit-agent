"""审核规则评测集
==================

把 ``finance/samples/`` 下的全部样本票跑一遍，逐条核对：

    这张票**应该**命中哪条规则？**实际**命中了哪条？

这解决的问题是：**你怎么知道这套规则真的在工作？**
写 17 条规则很容易，说清它们准不准才难。没有评测，"我的 agent 能识别异常"
就只是一句自夸；有了评测，它变成一个可复现的数字。

指标
----
- **规则命中率**：预期命中的规则里，实际命中的比例（召回）
- **误报数**：没预期命中却命中了的（精确度）
- **抽取字段正确数**：从 PDF 抽出的字段与样本定义是否一致
- **端到端完成率**：全部样本跑完无异常的比例

用法::

    ./.venv/Scripts/python.exe scripts/run_eval.py
    ./.venv/Scripts/python.exe scripts/run_eval.py --report eval/report.md

评测**完全离线**：走 pypdf 文本层，不调视觉模型，不调 LLM 叙述。
跑一百遍结果一样 —— 这本身就是"规则引擎零 LLM"带来的好处。
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finance import (  # noqa: E402
    ReimbursementRequest,
    Severity,
    parse_money,
)
from finance.audit import run_audit  # noqa: E402
from finance.store import AuditStore  # noqa: E402

SAMPLES_DIR = PROJECT_ROOT / "finance" / "samples"
_RE_RULE_ID = re.compile(r"R\d{3}")


@dataclass
class SampleOutcome:
    key: str
    title: str
    expected_rules: list[str]
    actual_non_pass: dict[str, str]      # rule_id -> severity
    suggested: str
    extraction_ok: bool
    extraction_notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def hits_expected(self) -> bool:
        """预期的规则都命中了（允许额外命中，单独算误报）。"""
        return all(r in self.actual_non_pass for r in self.expected_rules)

    @property
    def false_positives(self) -> list[str]:
        return [r for r in self.actual_non_pass if r not in self.expected_rules]


def load_manifest() -> dict:
    path = SAMPLES_DIR / "samples.yaml"
    if not path.is_file():
        raise SystemExit(
            f"找不到样本清单 {path}\n"
            "请先运行：./.venv/Scripts/python.exe scripts/make_samples.py"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_request(spec: dict) -> ReimbursementRequest:
    return ReimbursementRequest(
        applicant=spec.get("applicant", ""),
        department=spec.get("department", ""),
        expense_type=spec.get("expense_type", ""),
        amount=parse_money(spec.get("amount", "0")),
        reason=spec.get("reason", ""),
        submit_date=date.fromisoformat(spec["submit_date"]),
        city=spec.get("city") or "",
        nights=spec.get("nights"),
        headcount=spec.get("headcount"),
        has_itemized_list=bool(spec.get("has_itemized_list")),
    )


def check_extraction(invoice, spec: dict) -> tuple[bool, list[str]]:
    """核对抽取字段。

    只查能直接从样本清单推出的几项，**但必须覆盖规则真正依赖的字段**。
    曾经的版本漏掉了 invoice_type，于是「字段抽取正确率 100%」在发票类型
    全部抽成空串、13 张样本全被误杀的情况下依然显示 100% —— 指标成了摆设。
    """
    notes: list[str] = []
    req = spec["request"]

    if invoice.buyer_name and invoice.buyer_name != "示例科技有限公司":
        return True, ["S04 类样本：抬头本就不是公司全称，跳过核对"]
    if not invoice.invoice_number:
        notes.append("发票号码抽取为空")
    if not invoice.invoice_type:
        notes.append("发票类型抽取为空")
    if invoice.total is None:
        notes.append("价税合计抽取为空")
    elif parse_money(invoice.total) != parse_money(req.get("amount", 0)):
        notes.append(
            f"价税合计 {invoice.total} 与样本定义 {req.get('amount')} 不一致"
        )
    if invoice.issue_date is None:
        notes.append("开票日期抽取为空")
    return (not notes), notes


async def evaluate_all() -> list[SampleOutcome]:
    manifest = load_manifest()
    outcomes: list[SampleOutcome] = []

    with tempfile.TemporaryDirectory() as tmp:
        # 全部样本共用**同一个** store —— 连号检测（R012）依赖跨单历史，
        # 每单一个干净目录的话永远测不出连号。
        store = AuditStore(base_dir=tmp)

        for key, spec in manifest.items():
            expected = _RE_RULE_ID.findall(spec.get("expects", ""))
            pdf_path = SAMPLES_DIR / spec["pdf"]
            outcome = SampleOutcome(
                key=key,
                title=spec.get("title", ""),
                expected_rules=expected,
                actual_non_pass={},
                suggested="",
                extraction_ok=True,
            )
            try:
                result = await run_audit(
                    pdf_path,
                    make_request(spec["request"]),
                    store=store,
                    use_vision=False,        # 评测走确定性路径，不联网
                    narrative_llm=None,      # 不用模型，模板叙述
                )
                outcome.suggested = result.suggested_status.value
                outcome.actual_non_pass = {
                    f.rule_id: f.severity.value
                    for f in result.findings
                    if f.severity is not Severity.PASS
                }
                outcome.extraction_ok, outcome.extraction_notes = check_extraction(
                    result.invoice, spec
                )
            except Exception as exc:  # noqa: BLE001
                outcome.error = f"{type(exc).__name__}: {exc}"

            outcomes.append(outcome)

    return outcomes


def render(outcomes: list[SampleOutcome]) -> tuple[str, dict]:
    """生成报告文本与指标。"""
    total = len(outcomes)
    errored = [o for o in outcomes if o.error]
    completed = total - len(errored)
    with_expectation = [o for o in outcomes if o.expected_rules]

    expected_total = sum(len(o.expected_rules) for o in with_expectation)
    expected_hit = sum(
        len([r for r in o.expected_rules if r in o.actual_non_pass])
        for o in with_expectation
    )
    fp_total = sum(len(o.false_positives) for o in outcomes)
    extraction_ok = sum(1 for o in outcomes if o.extraction_ok and not o.error)

    recall = (expected_hit / expected_total * 100) if expected_total else 100.0
    completion = (completed / total * 100) if total else 0.0
    extraction_rate = (extraction_ok / total * 100) if total else 0.0

    lines: list[str] = []
    lines.append("# 报销审核规则 —— 评测报告")
    lines.append("")
    lines.append("> 由 `scripts/run_eval.py` 自动生成。完全离线：pypdf 文本层 + 规则引擎，")
    lines.append("> 不调视觉模型、不调 LLM。同一批样本跑一百遍结果一致。")
    lines.append("")
    lines.append("## 指标")
    lines.append("")
    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| 端到端完成率 | {completion:.1f}% | {completed}/{total} 张样本跑完无异常 |"
    )
    lines.append(
        f"| 规则召回率 | {recall:.1f}% | 预期命中的 {expected_total} 条规则中，"
        f"实际命中 {expected_hit} 条 |"
    )
    lines.append(f"| 误报数 | {fp_total} | 未预期命中却命中的规则条数 |")
    lines.append(
        f"| 字段抽取正确率 | {extraction_rate:.1f}% | {extraction_ok}/{total} 张样本"
        "抽取字段与样本定义一致 |"
    )
    lines.append("")
    lines.append("## 逐样本明细")
    lines.append("")
    lines.append("| 样本 | 预期命中 | 实际非通过项 | 系统建议 | 判定 |")
    lines.append("|---|---|---|---|---|")

    for o in outcomes:
        if o.error:
            lines.append(f"| {o.key} | {'/'.join(o.expected_rules) or '—'} | — | — | ❌ 异常 |")
            continue
        expected = "/".join(o.expected_rules) or "（全通过）"
        actual = "、".join(f"{k}={v}" for k, v in o.actual_non_pass.items()) or "（全通过）"
        if not o.hits_expected:
            verdict = "❌ 漏报"
        elif o.false_positives:
            verdict = "⚠️ 有误报"
        else:
            verdict = "✅"
        lines.append(f"| {o.key} | {expected} | {actual} | {o.suggested} | {verdict} |")

    if errored:
        lines.append("")
        lines.append("## 异常明细")
        lines.append("")
        for o in errored:
            lines.append(f"- **{o.key}**：{o.error}")

    lines.append("")
    lines.append("## 样本说明")
    lines.append("")
    for o in outcomes:
        lines.append(f"- `{o.key}` —— {o.title}")
        if o.extraction_notes:
            for note in o.extraction_notes:
                lines.append(f"  - 抽取提示：{note}")

    metrics = {
        "total": total,
        "completion": completion,
        "recall": recall,
        "false_positives": fp_total,
        "extraction_rate": extraction_rate,
        "failed": [o.key for o in outcomes if not o.hits_expected or o.error],
    }
    return "\n".join(lines) + "\n", metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="报销审核规则评测")
    parser.add_argument("--report", type=str, default="", help="把报告写到指定 Markdown 文件")
    args = parser.parse_args()

    outcomes = asyncio.run(evaluate_all())
    report, metrics = render(outcomes)

    print(report)

    if args.report:
        path = Path(args.report)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"\n报告已写入 -> {path}")

    ok = not metrics["failed"]
    print()
    print("=" * 56)
    print("评测结论：" + ("全部样本符合预期 ✅" if ok else f"存在偏差 ❌ {metrics['failed']}"))
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
