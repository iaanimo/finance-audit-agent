"""三路对比：规则引擎 vs 大模型直判
====================================

这个脚本只做一件容易被说漂亮、但必须被量出来的事：

**同一批 27 张样本，把「判定权交给大模型」会怎样？**

对照组是 ``run_audit(narrative_llm=None)`` —— 19 条纯 Python 规则的判定，
端到端 27/27、误报 0（见 ``eval/report.md``）。实验组是把同样的输入喂给模型，
让它直接下结论。

为了公平，喂给模型的**不是**规则清单，而是规则所依据的**公司制度要点**
（限额、时限、抬头、税率、票种）。也就是说：模型拿到的信息量不比规则引擎少，
差别只在「谁来做判断」——这才能回答"能不能让模型判"。

三组指标，缺一不可：

1. **判定一致率**：模型结论与规则结论之比。看的是"准不准"。
2. **可复现性**：同一张票跑 3 次，结论会不会变。看的是"稳不稳"——
   财务审核里"这次驳、下次过"比"一直判错"更不可接受。
3. **错误方向**：把违规单放行（false approve）和把合规单驳回（false reject）
   代价完全不同，混在一个"准确率"里就等于没说。

第三路（小模型 SFT 后判定）需要 GPU 与微调环境，尚未跑：
报告里保留占位，不编数字。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance.audit import run_audit  # noqa: E402
from finance.models import ReimbursementRequest  # noqa: E402
from finance.store import AuditStore  # noqa: E402

SAMPLE_DIR = PROJECT_ROOT / "finance" / "samples"

POLICY_BRIEF = """公司报销制度要点：
1. 发票抬头必须为公司全称「示例科技有限公司」，纳税人识别号须一致；
2. 报销时限：开票日距提交日不超过 60 天，且不得跨年报销；
3. 市内交通费单次不超过 200 元；餐饮费人均不超过 150 元且须注明用餐人数；
4. 住宿费：一线城市每晚不超过 600 元，其他城市每晚不超过 400 元；
5. 办公用品单张发票超过 2000 元须附采购清单；
6. 发票类型：定额发票不接受；境外票据须人工核验汇率；
7. 金额勾稽：申请金额必须等于发票价税合计；不含税金额 + 税额 必须等于价税合计；
   大写与小写金额必须一致；
8. 税率须与项目相符（餐饮住宿、技术服务等 6%，简易计税 3%）；
9. 费用类型须能对应到会计科目；部门须在预算表内。"""

VERDICT_MAP = {"通过": "APPROVED", "驳回": "REJECTED", "转人工": "PENDING"}


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


def build_prompt(spec: dict, invoice) -> str:
    req = spec.get("request") or {}
    inv = spec.get("invoice") or {}
    return "\n".join(
        [
            "你是企业财务共享中心的报销审核员。依据下面的公司制度，判断这张报销单应当",
            "「通过」「驳回」还是「转人工」（信息不足、需人工核实时选转人工）。",
            "",
            POLICY_BRIEF,
            "",
            "=== 发票票面 ===",
            f"发票号码：{inv.get('invoice_number', '')}",
            f"发票类型：{inv.get('invoice_type', '')}",
            f"开票日期：{inv.get('issue_date', '')}",
            f"购买方：{inv.get('buyer_name', '')}",
            f"销售方：{inv.get('seller_name', '')}",
            f"项目名称：{inv.get('item_name', '')}",
            f"价税合计：{inv.get('total', '')}",
            f"（抽取结果）不含税金额：{invoice.amount}　税额：{invoice.tax_amount}　"
            f"税率：{invoice.tax_rate}　大写：{invoice.total_in_words or '（未抽取到）'}",
            f"备注：{invoice.remark or '（空）'}",
            "",
            "=== 报销申请 ===",
            f"申请人：{req.get('applicant', '')}　部门：{req.get('department', '')}",
            f"费用类型：{req.get('expense_type', '')}　申请金额：{req.get('amount', '')}",
            f"事由：{req.get('reason', '')}　提交日期：{req.get('submit_date', '')}",
            f"城市：{req.get('city') or '—'}　住宿晚数：{req.get('nights')}　"
            f"用餐人数：{req.get('headcount')}　附清单：{req.get('has_itemized_list')}",
            "",
            "只输出一个 JSON 对象，不要任何解释文字、不要代码块围栏：",
            '{"verdict": "通过|驳回|转人工", "reason": "不超过 60 字", '
            '"hit": "最像被违反的那一条制度要点（短名，无则填 无)"}',
        ]
    )


def parse_json(text: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象（容忍 ```json 围栏）。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def run(args: argparse.Namespace) -> int:
    from core.llm_factory import create_llm

    llm = create_llm(temperature=0.0, max_tokens=300)
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")

    manifest = yaml.safe_load((SAMPLE_DIR / "samples.yaml").read_text(encoding="utf-8"))
    keys = list(manifest)
    if args.limit:
        keys = keys[: args.limit]

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = AuditStore(tmp)
        for key in keys:
            spec = manifest[key]
            # 对照组：规则引擎（零 LLM），这就是 ground truth
            result = await run_audit(
                SAMPLE_DIR / spec["pdf"], build_request(spec), store=store, narrative_llm=None
            )
            truth = result.suggested_status.value
            prompt = build_prompt(spec, result.invoice)

            verdicts: list[str] = []
            for rep in range(args.repeat):
                t0 = time.perf_counter()
                try:
                    resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=60)
                    text = getattr(resp, "content", "") or ""
                    latency = time.perf_counter() - t0
                    usage = getattr(resp, "usage_metadata", None) or {}
                    tokens = int(usage.get("total_tokens") or 0)
                except Exception as e:  # noqa: BLE001
                    rows.append({"key": key, "rep": rep, "model": "error", "truth": truth,
                                 "reason": f"{type(e).__name__}: {e}", "latency": 0, "tokens": 0, "hit": ""})
                    continue

                data = parse_json(text) or {}
                raw = str(data.get("verdict", "")).strip()
                verdict = VERDICT_MAP.get(raw) or next(
                    (v for k, v in VERDICT_MAP.items() if k in raw), "PARSE_FAIL"
                )
                verdicts.append(verdict)
                rows.append({
                    "key": key, "rep": rep, "model": verdict, "truth": truth,
                    "reason": str(data.get("reason", ""))[:70],
                    "hit": str(data.get("hit", ""))[:24],
                    "latency": round(latency, 2), "tokens": tokens,
                })

            distinct = {v for v in verdicts}
            flag = "一致" if distinct == {truth} else ("不稳定" if len(distinct) > 1 else "不一致")
            print(f"  {key:<34} 真值 {truth:<9} 模型 {','.join(verdicts):<26} {flag}", flush=True)

    ok = [r for r in rows if r["model"] in ("APPROVED", "REJECTED", "PENDING")]
    n_calls = len(rows)
    agree = [r for r in ok if r["model"] == r["truth"]]
    false_approve = [r for r in ok if r["truth"] == "REJECTED" and r["model"] == "APPROVED"]
    false_reject = [r for r in ok if r["truth"] == "APPROVED" and r["model"] == "REJECTED"]
    parse_fail = [r for r in rows if r["model"] == "PARSE_FAIL"]

    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(r["key"], []).append(r)
    unstable = [k for k, rs in by_key.items() if len({r["model"] for r in rs if r["model"] != "error"}) > 1]
    stable_wrong = [k for k, rs in by_key.items()
                    if len({r["model"] for r in rs if r["model"] != "error"}) == 1
                    and rs[0]["model"] != rs[0]["truth"]]

    lat = sorted(r["latency"] for r in ok) or [0.0]
    p50 = lat[len(lat) // 2]
    tokens = sum(r["tokens"] for r in ok)

    lines = [
        "# 三路对比：规则引擎 vs 大模型直判",
        "",
        "> 由 `scripts/measure_llm_judge.py` 生成。对照组 = 19 条纯 Python 规则（零 LLM）；",
        "> 实验组 = 同一批样本交给模型直接判，喂给它的是**制度要点**而不是规则清单。",
        f"> 模型：`{model_name}`（temperature=0）　样本 {len(keys)} 张 × {args.repeat} 次 = {n_calls} 次调用",
        "",
        "## 总览",
        "",
        "| 路线 | 判定方式 | 样本 | 与规则一致 | 可复现 | 备注 |",
        "|---|---|---|---|---|---|",
        f"| ① 规则引擎 | 19 条纯 Python 规则，零 LLM | 27 | **27/27（100%）** | **100%** | 端到端评测 `eval/report.md` |",
        f"| ② 大模型直判 | {model_name}，temperature=0 | {len(keys)} | "
        f"**{len(agree)}/{len(ok)}（{_pct(len(agree), len(ok))}）** | "
        f"{_pct(len(by_key) - len(unstable), len(by_key))}（{len(unstable)} 张三次结论不一致） | 见下表 |",
        "| ③ 小模型 SFT 后判定 | 未跑 | — | — | — | **缺 GPU / 微调环境，不编数字** |",
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
        "- **一致率不是模型能力排名**：模型拿到的是制度要点，不是规则清单；差异反映的是",
        "  「把判断权交给概率模型」这件事本身的性质。",
        "- **可复现性才是关键指标**：财务审核里同一张票两次结论不同，比一直判错更不可接受 ——",
        "  后者能被发现并修正，前者连「以哪次为准」都说不清。",
        "- 结论只证明一件事：在本项目的 19 条规则所覆盖的场景上，规则引擎与模型直判存在可测量的差距。",
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
    parser.add_argument("--repeat", type=int, default=3, help="每张样本重复判定次数（默认 3，用于测稳定性）")
    parser.add_argument("--limit", type=int, default=0, help="只测前 N 张样本")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "eval" / "llm-judge-report.md"))
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
