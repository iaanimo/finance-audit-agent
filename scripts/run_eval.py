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
#: ``expects`` 里的一条期望：规则号，后面**紧跟**的严重度是可选的。
#: 只认紧跟的 —— 否则「R016 FAIL（大写 1560 ≠ 小写 1650）」后面那句里的字样
#: 会被粘到别的规则号上。
_RE_EXPECT = re.compile(r"(R\d{3})(?:\s*(PASS|WARN|FAIL))?")


@dataclass
class SampleOutcome:
    key: str
    title: str
    expected: dict[str, str | None]      # rule_id -> 期望严重度（None 表示只查命中）
    actual_non_pass: dict[str, str]      # rule_id -> 实际严重度
    suggested: str
    extraction_ok: bool
    extraction_notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def expected_rules(self) -> list[str]:
        return list(self.expected)

    @property
    def severity_mismatch(self) -> list[str]:
        """命中了、但严重度与 ``expects`` 写的不一样。

        只查"命中没命中"是不够的：把 R007 从 FAIL 降级成 WARN，
        过去照样算"命中"—— 而 FAIL 和 WARN 在状态机里一个走向驳回、
        一个走向转人工，是**完全不同的结论**。
        """
        out = []
        for rule_id, want in self.expected.items():
            if want is None:
                continue
            got = self.actual_non_pass.get(rule_id)
            if got is not None and got != want:
                out.append(f"{rule_id} 期望 {want}、实际 {got}")
        return out

    @property
    def hits_expected(self) -> bool:
        """预期的规则都命中，且严重度对得上（允许额外命中，单独算误报）。"""
        return all(r in self.actual_non_pass for r in self.expected) and not self.severity_mismatch

    @property
    def false_positives(self) -> list[str]:
        return [r for r in self.actual_non_pass if r not in self.expected]

    @property
    def is_bad(self) -> bool:
        """这张样本算不算失败。

        **误报也算失败。** 过去它只在表格里挂个 ⚠️，不进 ``failed`` ——
        于是 S01 误报 2 条时脚本照样打印「全部样本符合预期 ✅」并返回 0。
        一个"只报喜不报忧"的评测等于没有评测。
        """
        return bool(self.error) or not self.hits_expected or bool(self.false_positives)


def load_manifest() -> dict:
    path = SAMPLES_DIR / "samples.yaml"
    if not path.is_file():
        raise SystemExit(
            f"找不到样本清单 {path}\n"
            "请先运行：./.venv/Scripts/python.exe scripts/make_samples.py"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_expects(text: str) -> dict[str, str | None]:
    """把 ``expects`` 那句人话解析成 ``{规则号: 期望严重度}``。"""
    return {rid: sev for rid, sev in _RE_EXPECT.findall(text or "")}


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
    """逐字段核对抽取结果与样本清单里**显式写出**的票面值。

    过去这里有个启发式：「抬头非空且不等于公司全称 → 跳过全部核对」，
    本意是放过 S04（抬头本来就是"个人"）。它的问题是**把一条特例变成了通用开关**：
    任何一张票只要抬头被抽错，整张票的字段核对全部跳过，指标照样满分。
    这正是这个函数自己的注释里说"已经修好"的那个洞换了个字段重演 ——
    所以现在不猜了，样本清单直接写明每一张票的票面值是什么。
    """
    notes: list[str] = []
    want: dict = spec.get("invoice") or {}
    if not want:
        return False, ["样本清单里没有 invoice 段，无法核对抽取字段"]

    for field, label, is_money in (
        ("invoice_number", "发票号码", False),
        ("invoice_type", "发票类型", False),
        ("buyer_name", "购买方名称", False),
        ("seller_name", "销售方名称", False),
        ("item_name", "项目名称", False),
        ("total", "价税合计", True),
    ):
        if field not in want:
            continue
        expected = want[field]
        actual = getattr(invoice, field)
        if actual is None or actual == "":
            notes.append(f"{label}抽取为空（样本定义 {expected!r}）")
            continue
        if is_money:
            if parse_money(actual) != parse_money(expected):
                notes.append(f"{label} {actual} 与样本定义 {expected} 不一致")
        elif str(actual).strip() != str(expected).strip():
            notes.append(f"{label}「{actual}」与样本定义「{expected}」不一致")

    if "issue_date" in want:
        want_date = date.fromisoformat(str(want["issue_date"]))
        if invoice.issue_date is None:
            notes.append(f"开票日期抽取为空（样本定义 {want_date.isoformat()}）")
        elif invoice.issue_date != want_date:
            notes.append(f"开票日期 {invoice.issue_date} 与样本定义 {want_date} 不一致")

    return (not notes), notes


async def evaluate_all() -> list[SampleOutcome]:
    manifest = load_manifest()
    outcomes: list[SampleOutcome] = []

    with tempfile.TemporaryDirectory() as tmp:
        # 全部样本共用**同一个** store —— 连号检测（R012）依赖跨单历史，
        # 每单一个干净目录的话永远测不出连号。
        store = AuditStore(base_dir=tmp)

        for key, spec in manifest.items():
            pdf_path = SAMPLES_DIR / spec["pdf"]
            outcome = SampleOutcome(
                key=key,
                title=spec.get("title", ""),
                expected=parse_expects(spec.get("expects", "")),
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

    # 分母是**期望实例**不是规则条数：同一条规则可能出现在多张样本上
    # （R012 连号就有两张），按"规则条数"报会把 12 说成 11。
    expected_total = sum(len(o.expected_rules) for o in with_expectation)
    expected_hit = sum(
        len([r for r in o.expected_rules if r in o.actual_non_pass])
        for o in with_expectation
    )
    distinct_expected = {r for o in with_expectation for r in o.expected_rules}
    fp_total = sum(len(o.false_positives) for o in outcomes)
    extraction_ok = sum(1 for o in outcomes if o.extraction_ok and not o.error)

    severity_mismatches = [m for o in outcomes for m in o.severity_mismatch]
    extraction_total = sum(1 for o in outcomes if not o.error)
    recall = (expected_hit / expected_total * 100) if expected_total else 100.0
    completion = (completed / total * 100) if total else 0.0
    extraction_rate = (
        (extraction_ok / extraction_total * 100) if extraction_total else 0.0
    )

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
        f"| 规则召回率 | {recall:.1f}% | 预期命中的 {expected_total} 个**期望实例**中，"
        f"实际命中 {expected_hit} 个（覆盖 {len(distinct_expected)} 条不同规则） |"
    )
    lines.append(
        f"| 误报数 | {fp_total} | 未预期命中却命中的规则条数（**计入失败**） |"
    )
    lines.append(
        f"| 严重度一致 | {len(severity_mismatches)} | 命中但严重度与预期不符的条数"
        "（FAIL 与 WARN 走向完全不同的结论） |"
    )
    lines.append(
        f"| 字段抽取正确率 | {extraction_rate:.1f}% | {extraction_ok}/{extraction_total} "
        "张跑完的样本抽取字段与样本清单一致 |"
    )
    lines.append("")
    lines.append("## 逐样本明细")
    lines.append("")
    lines.append("| 样本 | 预期命中 | 实际非通过项 | 系统建议 | 判定 |")
    lines.append("|---|---|---|---|---|")

    for o in outcomes:
        expected = (
            "、".join(f"{r}={s}" if s else r for r, s in o.expected.items()) or "（全通过）"
        )
        if o.error:
            lines.append(f"| {o.key} | {expected} | — | — | ❌ 异常 |")
            continue
        actual = "、".join(f"{k}={v}" for k, v in o.actual_non_pass.items()) or "（全通过）"
        if not o.hits_expected:
            verdict = "❌ 与预期不符"
        elif o.false_positives:
            verdict = "⚠️ 有误报（计入失败）"
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
        "severity_mismatches": severity_mismatches,
        "extraction_rate": extraction_rate,
        # 误报、严重度不符、抽取不符，全部计入失败 —— 退出码要能反映它们
        "failed": [o.key for o in outcomes if o.is_bad or not o.extraction_ok],
    }
    return "\n".join(lines) + "\n", metrics


def main() -> int:
    # 报告含 ✅/❌ 等非 GBK 字符，Windows 默认控制台（cp936）print 时会抛
    # UnicodeEncodeError，整次评测在打印阶段崩掉、报告都来不及写。
    # 统一按 UTF-8 输出 —— 评测是给人看的，不能死在展示上。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

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
    if ok:
        print(f"评测结论：全部 {metrics['total']} 张样本符合预期 ✅")
    else:
        print(f"评测结论：存在偏差 ❌ {metrics['failed']}")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
