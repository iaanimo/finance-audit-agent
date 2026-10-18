"""判定提示词：**唯一来源**
========================

同一个模型判定任务，有三个地方要用提示词：

1. ``scripts/measure_llm_judge.py`` —— 拿通用大模型做基线，量它和规则的差距；
2. ``scripts/build_sft_data.py`` —— 造微调数据时，输入必须和评测时**逐字一致**；
3. 将来接本地微调模型评测（同一脚本换 base_url）—— 还是这份。

如果三处各写一份，训练/评测的提示词就会漂，微调出来的模型在评测集上的分数
就不作数（模型学到的是"另一种问法"）。所以这里只留一个实现，配一条测试
断言两条路径生成的提示词完全相同。

**注意**：提示词里给的是**制度要点**，不是规则清单。给规则清单等于把引擎的
判据直接抄给模型，测出来的是"抄得像不像"，不是"会不会判"。
"""

from __future__ import annotations

import json
import re

from .models import Invoice, ReimbursementRequest

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
REVERSE_VERDICT = {v: k for k, v in VERDICT_MAP.items()}

OUTPUT_SCHEMA_HINT = (
    "只输出一个 JSON 对象，不要任何解释文字、不要代码块围栏：\n"
    '{"verdict": "通过|驳回|转人工", "reason": "不超过 60 字", '
    '"hit": "最像被违反的那一条制度要点（短名，无则填 无)"}'
)


def build_judge_prompt(invoice: Invoice, request: ReimbursementRequest) -> str:
    """构造判定提示词。**训练与评测共用这一份**。"""
    return "\n".join(
        [
            "你是企业财务共享中心的报销审核员。依据下面的公司制度，判断这张报销单应当",
            "「通过」「驳回」还是「转人工」（信息不足、需人工核实时选转人工）。",
            "",
            POLICY_BRIEF,
            "",
            "=== 发票票面 ===",
            f"发票号码：{invoice.invoice_number}",
            f"发票类型：{invoice.invoice_type}",
            f"开票日期：{invoice.issue_date.isoformat() if invoice.issue_date else ''}",
            f"购买方：{invoice.buyer_name}",
            f"销售方：{invoice.seller_name}",
            f"项目名称：{invoice.item_name}",
            f"价税合计：{invoice.total}",
            f"（抽取结果）不含税金额：{invoice.amount}　税额：{invoice.tax_amount}　"
            f"税率：{invoice.tax_rate}　大写：{invoice.total_in_words or '（未抽取到）'}",
            f"备注：{invoice.remark or '（空）'}",
            "",
            "=== 报销申请 ===",
            f"申请人：{request.applicant}　部门：{request.department}",
            f"费用类型：{request.expense_type}　申请金额：{request.amount}",
            f"事由：{request.reason}　提交日期：{request.submit_date}",
            f"城市：{request.city or '—'}　住宿晚数：{request.nights}　"
            f"用餐人数：{request.headcount}　附清单：{request.has_itemized_list}",
            "",
            OUTPUT_SCHEMA_HINT,
        ]
    )


def extract_json(text: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象（容忍 ```json 围栏与前后废话）。"""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def to_verdict(raw: str) -> str:
    """把模型输出的中文结论映射成系统状态；认不出来就返回 PARSE_FAIL。"""
    raw = (raw or "").strip()
    if raw in VERDICT_MAP:
        return VERDICT_MAP[raw]
    for zh, code in VERDICT_MAP.items():
        if zh in raw:
            return code
    return "PARSE_FAIL"
