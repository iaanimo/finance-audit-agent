"""护栏拦截率实测
==================

**这张报告回答的问题，和 ``run_eval.py`` 回答的不是同一个。**

``run_eval.py`` 证明「规则的判定是确定性的」—— 跑一百遍结果一致。
但它证明不了另一件事：**叙述层那两道护栏真的在拦东西吗？**

「模型可能编数字」「模型可能把结论说反」是设计前提，过去只有单元测试
（手写一段坏叙述喂给 guard）。而单元测试只能证明「喂坏数据会被拦」，
证明不了「模型在真实提示词下有多经常产出坏数据」—— 那个数字只能实测。

所以这个脚本做的是：

1. 用 ``narrative_llm=None`` 跑完整审核，拿到**零 LLM 的判定结果**；
2. 拿同一份 findings 构造真实提示词，**直接调模型**（绕过 audit.py 里的
   兜底 try/except，否则失败会被静默吞掉，看不到）；
3. 把模型输出分别过两道护栏，逐条记录：通过 / 编数字被拦 / 结论相反被拦；
4. 汇总：通过率、拦截率、拦截原因分布、耗时分位、token 消耗。

口径说明（免得把数字读高）：

- 拦截**不是**坏事。拦一次 = 一段可能误导审核员的话没有进入正式单据。
- 通过率受提示词质量影响，不是模型能力的排名，本脚本只记录事实。
- 全程写临时目录，**不碰 data/**，也不产生审核单。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance.audit import NARRATIVE_TIMEOUT_SECONDS  # noqa: E402
from finance.audit import _build_narrative_prompt, run_audit  # noqa: E402
from finance.guard import (  # noqa: E402
    aggregate,
    stated_verdict,
    unverifiable_numbers,
)
from finance.models import ReimbursementRequest  # noqa: E402
from finance.store import AuditStore  # noqa: E402

SAMPLE_DIR = PROJECT_ROOT / "finance" / "samples"


def build_request(spec: dict) -> ReimbursementRequest:
    """把清单里的 ``request`` 段还原成报销申请单。"""
    req = spec.get("request") or {}
    return ReimbursementRequest(
        applicant=req.get("applicant", ""),
        department=req.get("department", ""),
        expense_type=req.get("expense_type", ""),
        amount=str(req.get("amount") or "0"),
        reason=req.get("reason", ""),
        submit_date=date.fromisoformat(req.get("submit_date") or date.today().isoformat()),
        city=req.get("city", "") or "",
        nights=req.get("nights"),
        headcount=req.get("headcount"),
        has_itemized_list=bool(req.get("has_itemized_list")),
    )


def classify(text: str, findings) -> tuple[str, str]:
    """按护栏的两道关分别判定，返回 (结论, 原因说明)。

    结论取值：``pass`` / ``blocked:number`` / ``blocked:verdict`` / ``empty``
    """
    if not text or not text.strip():
        return "empty", "模型返回空文本"

    offending = unverifiable_numbers(text, findings)
    if offending:
        return "blocked:number", "编造数字：" + "、".join(offending[:5])

    stated = stated_verdict(text)
    expected = aggregate(findings).value
    if stated is not None and stated != expected:
        return "blocked:verdict", f"结论倾向 {stated} ≠ 系统建议 {expected}"

    return "pass", ""


async def run(args: argparse.Namespace) -> int:
    from core.llm_factory import create_llm

    llm = create_llm(temperature=0.0, max_tokens=400)
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")

    manifest = yaml.safe_load((SAMPLE_DIR / "samples.yaml").read_text(encoding="utf-8"))
    keys = list(manifest)
    if args.limit:
        keys = keys[: args.limit]

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = AuditStore(tmp)
        for repeat in range(args.repeat):
            for key in keys:
                spec = manifest[key]
                try:
                    result = await run_audit(
                        SAMPLE_DIR / spec["pdf"],
                        build_request(spec),
                        store=store,
                        narrative_llm=None,          # 判定零 LLM，这是测量基线
                    )
                except Exception as e:  # noqa: BLE001
                    rows.append({"key": key, "repeat": repeat, "outcome": "audit_failed",
                                 "reason": f"{type(e).__name__}: {e}", "latency": 0.0, "tokens": 0})
                    continue

                try:
                    prompt = _build_narrative_prompt(result)
                    if args.no_constraint:
                        # 消融对照 C：把「只能引用已出现的数字」这条硬约束拿掉。
                        # 想知道的是：这条约束到底是护栏的前提，还是可有可无的客套话。
                        prompt = re.sub(
                            r"硬性要求：.*?4\. 如果有驳回项，要明确指出命中哪条制度条款。",
                            "要求：写一段 100 字以内的中文审核意见，说清问题即可。",
                            prompt,
                            flags=re.S,
                        )
                    if args.mask_numbers:
                        # 消融实验：拿掉一切数字，模型只能靠猜 —— 用来检验护栏
                        # 在"真实提示词 + 真实模型"下到底抓不抓得住编造，
                        # 而不只是抓得住单元测试里手写的那段坏叙述。
                        prompt = re.sub(r"[0-9]", "◼", prompt)
                except Exception as e:  # noqa: BLE001
                    # 提示词都构造不出来，等于"这张单子永远用不上模型"。
                    # 这类失败必须被记下来，不能被当成一次普通调用失败吞掉。
                    rows.append({"key": key, "repeat": repeat, "outcome": "prompt_failed",
                                 "reason": f"{type(e).__name__}: {e}", "latency": 0.0, "tokens": 0})
                    continue
                t0 = time.perf_counter()
                try:
                    resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=NARRATIVE_TIMEOUT_SECONDS)
                    text = getattr(resp, "content", "") or ""
                    latency = time.perf_counter() - t0
                    usage = getattr(resp, "usage_metadata", None) or {}
                    tokens = int(usage.get("total_tokens") or 0)
                except Exception as e:  # noqa: BLE001
                    rows.append({"key": key, "repeat": repeat, "outcome": "llm_error",
                                 "reason": f"{type(e).__name__}: {e}",
                                 "latency": time.perf_counter() - t0, "tokens": 0})
                    continue

                outcome, reason = classify(text, result.findings)
                rows.append(
                    {
                        "key": key,
                        "repeat": repeat,
                        "outcome": outcome,
                        "reason": reason,
                        "latency": round(latency, 2),
                        "tokens": tokens,
                        "suggested": result.suggested_status.value,
                        "chars": len(text),
                        "narrative": text.strip().replace("\n", " ")[:160],
                    }
                )
                flag = "通过" if outcome == "pass" else "拦截"
                print(f"  [{repeat + 1}/{args.repeat}] {key:<34} {flag:<4} {latency:5.2f}s {tokens:>4} tok  {reason}", flush=True)

    total = len(rows)
    failed = [r for r in rows if r["outcome"] in ("llm_error", "audit_failed", "prompt_failed")]
    graded = [r for r in rows if r["outcome"] not in ("llm_error", "audit_failed", "prompt_failed")]
    passed = [r for r in graded if r["outcome"] == "pass"]
    blocked = [r for r in graded if r["outcome"].startswith("blocked")]
    empty = [r for r in graded if r["outcome"] == "empty"]

    lat = [r["latency"] for r in graded if r["latency"]]
    p50 = statistics.median(lat) if lat else 0.0
    p95 = (statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat or [0.0]))
    tokens = sum(r["tokens"] for r in graded)

    report = [
        "# 叙述层护栏 —— 实测报告",
        "",
        "> 由 `scripts/measure_guard.py` 自动生成。判定环节零 LLM，本报告只测**叙述层**。",
        f"> 模型：`{model_name}`（temperature=0）　调用次数：{total}　"
        f"样本 {len(keys)} 张 × 重复 {args.repeat} 次",
        "",
        "## 指标",
        "",
        "| 指标 | 数值 | 说明 |",
        "|---|---|---|",
        f"| 调用成功 | {len(graded)}/{total} | 其余为调用异常/超时 |",
        f"| **护栏放行** | {len(passed)}（{_pct(len(passed), len(graded))}） | 数字可溯源且结论不反，原文入档 |",
        f"| **护栏拦截** | {len(blocked)}（{_pct(len(blocked), len(graded))}） | 整段换模板，**结论不受影响** |",
        f"| 　· 编造数字 | {len([r for r in blocked if r['outcome'] == 'blocked:number'])} | 叙述里出现 findings 里没有的数字 |",
        f"| 　· 结论相反 | {len([r for r in blocked if r['outcome'] == 'blocked:verdict'])} | 模型下的结论与系统建议不一致 |",
        f"| 　· 空文本 | {len(empty)} | 模型返回空 |",
        f"| 调用失败 | {len(failed)} | 失败也只影响文字，不影响结论 |",
        f"| 延迟 p50 / p95 | {p50:.2f}s / {p95:.2f}s | 单次生成耗时 |",
        f"| token 合计 | {tokens} | 输入 + 输出 |",
        "",
        "## 逐次明细",
        "",
        "| 样本 | 次 | 结果 | 系统建议 | 耗时 | token | 原因 / 摘录 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["outcome"] in ("llm_error", "audit_failed", "prompt_failed"):
            report.append(f"| `{r['key']}` | {r['repeat'] + 1} | {r['outcome']} | — | "
                          f"{r['latency']:.2f}s | {r['tokens']} | {r['reason'][:60]} |")
            continue
        mark = {"pass": "放行", "empty": "空文本"}.get(r["outcome"], "拦截")
        detail = r["reason"] or r["narrative"][:70]
        report.append(
            f"| `{r['key']}` | {r['repeat'] + 1} | {mark} | {r['suggested']} | "
            f"{r['latency']:.2f}s | {r['tokens']} | {detail} |"
        )

    report += [
        "",
        "## 怎么读这张表",
        "",
        "- **拦截率高不是坏消息。** 每一次拦截都表示一段可能误导审核员的话没有进入正式单据；",
        "  拦截后由模板兜底，结论、状态、留痕都不变（`NarrativeSource.TEMPLATE`）。",
        "- **放行不代表模型可信**，只代表这一次它的输出恰好满足两道硬约束。",
        "  判定权始终在规则引擎，模型从来没有判定权 —— 这是本项目的前提，不是这里的结论。",
        "- 样本量小（百次量级），足够说明「护栏在真实提示词下会被触发」，不足以做模型横向比较。",
        "",
    ]

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")

    print()
    print("=" * 56)
    print(f"调用 {total} 次：放行 {len(passed)}，拦截 {len(blocked)}，失败 {len(failed)}")
    print(f"拦截原因：编数字 {len([r for r in blocked if r['outcome'] == 'blocked:number'])}，"
          f"结论相反 {len([r for r in blocked if r['outcome'] == 'blocked:verdict'])}")
    print(f"延迟 p50 {p50:.2f}s / p95 {p95:.2f}s，token 合计 {tokens}")
    print(f"报告 -> {out}")
    return 0


def _pct(a: int, b: int) -> str:
    return f"{(100.0 * a / b):.1f}%" if b else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description="实测叙述层护栏的拦截率")
    parser.add_argument("--repeat", type=int, default=2, help="每张样本重复调用次数（默认 2）")
    parser.add_argument("--limit", type=int, default=0, help="只测前 N 张样本（0 = 全部）")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "eval" / "guard-report.md"))
    parser.add_argument(
        "--mask-numbers",
        action="store_true",
        help="消融对照：把提示词里的数字全部遮掉，观察模型在拿不到数字时会不会编（不是正常用法）",
    )
    parser.add_argument(
        "--no-constraint",
        action="store_true",
        help="消融对照：拿掉「只能引用已出现数字」这条硬约束，看模型会不会自己算/编",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
