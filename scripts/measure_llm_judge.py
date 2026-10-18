"""三路对比：规则引擎 vs 大模型直判（可指向本地微调模型）
=========================================================

这个脚本只做一件容易被说漂亮、但必须被量出来的事：

**同一批 27 张样本，把「判定权交给大模型」会怎样？**

对照组是 ``run_audit(narrative_llm=None)`` —— 19 条纯 Python 规则的判定，
端到端 27/27、误报 0（见 ``eval/report.md``）。实验组把同样的输入喂给模型，
让它直接下结论。提示词由 ``finance/judge_prompt.py`` 统一提供 —— 与
``scripts/build_sft_data.py`` 造训练数据时用的是**同一份**，保证训练/评测同分布。

三组指标，缺一不可：

1. **判定一致率**：模型结论与规则结论之比。看的是"准不准"。
2. **可复现性**：同一张票跑 3 次，结论会不会变。看的是"稳不稳" ——
   财务审核里"这次驳、下次过"比"一直判错"更不可接受。
3. **错误方向**：把违规单放行（false approve）和把合规单驳回（false reject）
   代价完全不同，混在一个"准确率"里就等于没说。

用法（第三路 = 微调后的小模型，通过 OpenAI 兼容端点接入）：

    # 通用大模型基线（走 .env 里的 DEEPSEEK_*）
    python scripts/measure_llm_judge.py --repeat 3

    # 本地微调模型（vLLM / llama.cpp / Ollama 的 OpenAI 兼容端口）
    python scripts/measure_llm_judge.py --base-url http://127.0.0.1:8000/v1 \
        --model qwen2.5-1.5b-lora-audit --api-key dummy --repeat 3 \
        --report eval/llm-judge-sft-report.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance.audit import run_audit  # noqa: E402
from finance.judge_prompt import (  # noqa: E402
    build_judge_prompt,
    extract_json,
    to_verdict,
)
from finance.models import ReimbursementRequest  # noqa: E402
from finance.store import AuditStore  # noqa: E402

SAMPLE_DIR = PROJECT_ROOT / "finance" / "samples"


def build_request(spec: dict) -> ReimbursementRequest:
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


async def run(args: argparse.Namespace) -> int:
    from core.llm_factory import create_llm

    llm = create_llm(
        model=args.model or None,
        temperature=0.0,
        max_tokens=300,
        base_url=args.base_url or None,
        api_key=args.api_key or None,
    )
    model_name = args.model or getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")

    manifest = yaml.safe_load((SAMPLE_DIR / "samples.yaml").read_text(encoding="utf-8"))
    keys = list(manifest)
    if args.limit:
        keys = keys[: args.limit]

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = AuditStore(tmp)
        for key in keys:
            spec = manifest[key]
            # 对照组：规则引擎（零 LLM），ground truth
            result = await run_audit(
                SAMPLE_DIR / spec["pdf"], build_request(spec), store=store, narrative_llm=None
            )
            truth = result.suggested_status.value
            prompt = build_judge_prompt(result.invoice, result.request)

            verdicts: list[str] = []
            for rep in range(args.repeat):
                t0 = time.perf_counter()
                try:
                    resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=args.timeout)
                    text = getattr(resp, "content", "") or ""
                    latency = time.perf_counter() - t0
                    usage = getattr(resp, "usage_metadata", None) or {}
                    tokens = int(usage.get("total_tokens") or 0)
                except Exception as e:  # noqa: BLE001
                    rows.append({"key": key, "rep": rep, "model": "error", "truth": truth,
                                 "reason": f"{type(e).__name__}: {e}", "latency": 0,
                                 "tokens": 0, "hit": ""})
                    verdicts.append("error")
                    continue

                data = extract_json(text) or {}
                verdict = to_verdict(str(data.get("verdict", "")))
                verdicts.append(verdict)
                rows.append({
                    "key": key, "rep": rep, "model": verdict, "truth": truth,
                    "reason": str(data.get("reason", ""))[:70],
                    "hit": str(data.get("hit", ""))[:24],
                    "latency": round(latency, 2), "tokens": tokens,
                })

            distinct = {v for v in verdicts if v != "error"}
            flag = "一致" if distinct == {truth} else ("不稳定" if len(distinct) > 1 else "不一致")
            print(f"  {key:<34} 真值 {truth:<9} 模型 {','.join(verdicts):<26} {flag}", flush=True)

    ok = [r for r in rows if r["model"] in ("APPROVED", "REJECTED", "PENDING")]
    n_calls = len(rows)
    agree = [r for r in ok if r["model"] == r["truth"]]
    false_approve = [r for r in ok if r["truth"] == "REJECTED" and r["model"] == "APPROVED"]
    false_reject = [r for r in ok if r["truth"] == "APPROVED" and r["model"] == "REJECTED"]
    parse_fail = [r for r in rows if r["model"] == "PARSE_FAIL"]

    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[r["key"]].append(r)
    unstable = [k for k, rs in by_key.items()
                if len({r["model"] for r in rs if r["model"] != "error"}) > 1]

    lat = sorted(r["latency"] for r in ok) or [0.0]
    p50 = lat[len(lat) // 2]
    tokens = sum(r["tokens"] for r in ok)

    lines = [
        "# 三路对比：规则引擎 vs 大模型直判",
        "",
        "> 由 `scripts/measure_llm_judge.py` 生成。对照组 = 19 条纯 Python 规则（零 LLM）；",
        "> 实验组 = 同一批样本交给模型直接判，提示词由 `finance/judge_prompt.py` 统一提供",
        "> （给的是**制度要点**，不是规则清单 —— 信息量不打折，差别只在谁判）。",
        f"> 模型：`{model_name}`（temperature=0）　样本 {len(keys)} 张 × {args.repeat} 次 = {n_calls} 次调用",
        "",
        "## 总览",
        "",
        "| 路线 | 判定方式 | 样本 | 与规则一致 | 可复现 | 备注 |",
        "|---|---|---|---|---|---|",
        "| ① 规则引擎 | 19 条纯 Python 规则，零 LLM | 27 | **27/27（100%）** | **100%** | `eval/report.md` |",
        f"| ② 模型直判 | {model_name} | {len(keys)} | "
        f"**{len(agree)}/{len(ok)}（{_pct(len(agree), len(ok))}）** | "
        f"{_pct(len(by_key) - len(unstable), len(by_key))}（{len(unstable)} 张多次结论不一致） | 见下表 |",
        "",
        "## 错误方向（比准确率更重要）",
        "",
        "| 类型 | 次数 | 含义 |",
        "|---|---|---|",
        f"| 违规单被放行（false approve） | **{len(false_approve)}** | 最危险：不合规的费用进了账 |",
        f"| 合规单被驳回（false reject） | **{len(false_reject)}** | 代价是效率与体验，但不会造成错账 |",
        f"| 结论解析失败 | {len(parse_fail)} | 模型没按 JSON 输出 |",
        "",
        f"延迟 p50 {p50:.2f}s，token 合计 {tokens}。",
        "",
        "## 逐样本明细",
        "",
        "| 样本 | 规则真值 | 模型（每次） | 一致 | 模型理由 |",
        "|---|---|---|---|---|",
    ]
    for key in keys:
        rs = by_key.get(key, [])
        if not rs:
            continue
        truth = rs[0]["truth"]
        verdicts = "、".join(r["model"] for r in rs)
        same = "✅" if all(r["model"] == truth for r in rs) else (
            "⚠️" if len({r["model"] for r in rs}) > 1 else "❌"
        )
        lines.append(f"| `{key}` | {truth} | {verdicts} | {same} | {rs[0]['reason']} |")

    lines += [
        "",
        "## 怎么读",
        "",
        "- **一致率不是模型能力排名**：模型拿到的是制度要点，不是规则清单。",
        "- **可复现性才是关键指标**：同一张票两次结论不同，比一直判错更不可接受 ——",
        "  后者能被发现并修正，前者连「以哪次为准」都说不清。",
        "- 结论只证明：在 19 条规则覆盖的场景上，规则引擎与模型直判存在可测量的差距。",
        "",
    ]

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 56)
    print(f"规则 27/27；模型 {len(agree)}/{len(ok)} 一致（{_pct(len(agree), len(ok))}）")
    print(f"违规被放行 {len(false_approve)} 次，合规被驳回 {len(false_reject)} 次，"
          f"解析失败 {len(parse_fail)} 次；不稳定样本 {len(unstable)} 张")
    print(f"报告 -> {out}")
    return 0


def _pct(a: int, b: int) -> str:
    return f"{(100.0 * a / b):.1f}%" if b else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description="规则引擎 vs 大模型直判")
    parser.add_argument("--repeat", type=int, default=3, help="每张样本重复判定次数（默认 3）")
    parser.add_argument("--limit", type=int, default=0, help="只测前 N 张样本")
    parser.add_argument("--model", default="", help="模型名（默认取 .env 的 DEEPSEEK_MODEL）")
    parser.add_argument("--base-url", default="", help="OpenAI 兼容端点，如 http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="", help="本地端点通常填 dummy")
    parser.add_argument("--timeout", type=float, default=60.0, help="单次调用超时（秒）")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "eval" / "llm-judge-report.md"))
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
