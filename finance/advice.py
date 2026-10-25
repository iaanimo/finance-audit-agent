"""补正建议 —— 把"哪里不合规"翻译成"怎么改才合规"
=================================================

审核意见只说"驳回"没有用。报销人拿到驳回通知，第一件事是问：
**那我到底要改成什么？**

这个模块把每条命中规则的 ``evidence``（本来就有实际值/期望值）翻译成一句
可执行的话，例如：

- R007 住宿超标 -> 「每晚 800.00 元，需降至 600.00 元以内；3 晚合计从 2,400.00 降到 1,800.00」
- R010 金额不符 -> 「申请金额由 500.00 元改为 550.00 元（等于票面价税合计）」
- R003 超期     -> 「开票日距今 90 天，超过 60 天时限；超期无法补正，只能重开票或走特批」

三条硬约束（都有测试盯着）：

1. **只建议合规补正，不建议规避**。绝不给"拆成两张报""改个费用类型绕过去"这类
   出主意 —— 那是帮人绕过内控，不是帮人把单子改对。
2. **只读**。本模块不碰状态、不落库、不参与判定；它只是把已有结论换个说法。
3. **数字必须来自 evidence**。算出来的值（比如"降到多少"）由 evidence 里的
   限额与数量算出，不猜、不取整成好看的数字。
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

from .models import AuditFinding, Severity, parse_money

# 建议的性质，决定报销人下一步该干什么
KIND_FIX = "可补正"        # 改申请单/补材料就能过
KIND_REISSUE = "须重开票"   # 票面本身有问题，只能让销售方重开
KIND_MANUAL = "须人工/特批"  # 系统判不了，只能走人
KIND_SUPPLEMENT = "补充材料"  # 差一份材料

_SEVERITY_ORDER = {Severity.FAIL: 0, Severity.WARN: 1, Severity.PASS: 2}


def _money(value) -> str:
    """把 evidence 里的金额（float/str/Decimal）统一成两位小数的字符串。"""
    try:
        return f"{Decimal(str(value)):,.2f}"
    except Exception:  # noqa: BLE001
        return str(value)


def _num(value) -> Decimal | None:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return None


def _limit_from(expected) -> Decimal | None:
    """``expected`` 常写成 ``"<= 600.00"``，把里面的数抠出来。"""
    if isinstance(expected, (int, float, Decimal)):
        return _num(expected)
    text = str(expected or "")
    digits = "".join(ch for ch in text if ch.isdigit() or ch in ".")
    return _num(digits.rstrip(".")) if digits else None


def _advice(rule_id: str, clause: str, kind: str, text: str) -> dict:
    return {"rule_id": rule_id, "clause": clause, "kind": kind, "text": text}


def _for_r007(f: AuditFinding) -> dict:
    ev = f.evidence
    per_night = _num(ev.get("actual"))
    limit = _limit_from(ev.get("expected"))
    nights = ev.get("nights")
    if per_night is None or limit is None:
        return _advice("R007", f.clause, KIND_FIX, "住宿费超出所在城市限额，需按限额标准重报")
    text = f"每晚 {_money(per_night)} 元，需降至 {_money(limit)} 元以内"
    if nights:
        old_total = per_night * _num(nights)
        new_total = limit * _num(nights)
        text += f"；{nights} 晚合计由 {_money(old_total)} 元降到 {_money(new_total)} 元以内"
    return _advice("R007", f.clause, KIND_FIX, text + "（超出部分不予报销）")


def _for_r006(f: AuditFinding, meal_limit: Decimal = Decimal("150")) -> dict:
    ev = f.evidence
    total = _num(ev.get("invoice_total"))
    headcount = ev.get("actual")
    limit = meal_limit
    if headcount in (None, "", 0):
        if total is not None and limit > 0:
            need = (total / limit).to_integral_value(rounding=ROUND_CEILING)
            return _advice(
                "R006", f.clause, KIND_SUPPLEMENT,
                f"补注用餐人数（当前 {_money(total)} 元，人均需 ≤ {_money(limit)} 元，"
                f"人数至少 {int(need)} 人），否则无法核算人均",
            )
        return _advice("R006", f.clause, KIND_SUPPLEMENT, "补注用餐人数，以便核算人均金额")
    per = _num(headcount)
    if per and total is not None:
        allowed = limit * per
        return _advice(
            "R006", f.clause, KIND_FIX,
            f"{int(per)} 人人均 {_money(total / per)} 元，超过 {_money(limit)} 元/人；"
            f"金额需控制在 {_money(allowed)} 元以内（不得自行补人数凑标准）",
        )
    return _advice("R006", f.clause, KIND_FIX, "用餐人均金额超过标准，需按标准重报")


def _for_r010(f: AuditFinding) -> dict:
    ev = f.evidence
    claimed, total = _num(ev.get("claimed_amount")), _num(ev.get("invoice_total"))
    if claimed is not None and total is not None:
        return _advice(
            "R010", f.clause, KIND_FIX,
            f"申请金额由 {_money(claimed)} 元改为 {_money(total)} 元（须等于发票价税合计）",
        )
    return _advice("R010", f.clause, KIND_FIX, "申请金额需与发票价税合计一致")


def _for_r011(f: AuditFinding) -> dict:
    options = f.evidence.get("expected") or []
    hint = "、".join(options[:4]) if options else "正确的费用类型"
    return _advice(
        "R011", f.clause, KIND_FIX,
        f"费用类型「{f.evidence.get('actual', '')}」无法映射科目，请改为 {hint} 之一",
    )


def _for_r013(f: AuditFinding) -> dict:
    ev = f.evidence
    return _advice(
        "R013", f.clause, KIND_REISSUE,
        f"票面自相矛盾：不含税 {_money(ev.get('net_amount'))} + 税额 {_money(ev.get('tax_amount'))}"
        f" = {_money(ev.get('computed_total'))} 元，与价税合计 {_money(ev.get('total'))} 元不符；"
        "须由销售方更正后重开",
    )


def _for_r016(f: AuditFinding) -> dict:
    ev = f.evidence
    return _advice(
        "R016", f.clause, KIND_REISSUE,
        f"大写「{ev.get('actual', '')}」折算 {_money(ev.get('parsed_in_words'))} 元，"
        f"与小写 {_money(ev.get('expected'))} 元不一致；须由销售方重开",
    )


def _for_r017(f: AuditFinding) -> dict:
    ev = f.evidence
    allowed = "、".join(f"{r}%" for r in (ev.get("expected") or [])) or "制度适用税率"
    return _advice(
        "R017", f.clause, KIND_REISSUE,
        f"票面税率 {ev.get('actual', '')} 与项目「{ev.get('item_name', '')}」不符，"
        f"应为 {allowed}；须由销售方重开",
    )


def _for_r008(f: AuditFinding) -> dict:
    return _advice(
        "R008", f.clause, KIND_SUPPLEMENT,
        f"金额 {_money(f.evidence.get('actual'))} 元超过 2,000.00 元，须补附本次采购的明细清单"
        "（品名、单价、数量、领用部门）",
    )


def _for_r005(f: AuditFinding) -> dict:
    return _advice(
        "R005", f.clause, KIND_FIX,
        f"单次 {_money(f.evidence.get('actual'))} 元超过 {_money(_limit_from(f.evidence.get('expected')))} 元限额；"
        "超出部分不予报销（不得按行程拆成多笔报销）",
    )


def _for_r003(f: AuditFinding) -> dict:
    ev = f.evidence
    return _advice(
        "R003", f.clause, KIND_MANUAL,
        f"开票日 {ev.get('issue_date', '')} 距提交 {ev.get('submit_date', '')} 共 "
        f"{ev.get('age_days', '?')} 天，超过 {ev.get('expected', '60 天')}；"
        "超期无法通过补正解决，只能重开票或按特批流程处理",
    )


def _default(f: AuditFinding) -> dict:
    """没有专属建议的规则，给一句安全的话：说清依据，不给出主意。"""
    return _advice(
        f.rule_id, f.clause, KIND_MANUAL,
        f"{f.message}；该项由人工核实后决定（系统不代为判断）",
    )


BUILDERS = {
    "R003": _for_r003,
    "R005": _for_r005,
    "R006": _for_r006,
    "R007": _for_r007,
    "R008": _for_r008,
    "R010": _for_r010,
    "R011": _for_r011,
    "R013": _for_r013,
    "R016": _for_r016,
    "R017": _for_r017,
}

# 票面/账户层面的问题：只能重开票或走人，不属于"改申请单能解决"
FIXED_KIND = {
    "R001": (KIND_REISSUE, "发票抬头须为公司全称「{expected}」，须由销售方重开"),
    "R002": (KIND_REISSUE, "发票须列示公司纳税人识别号 {expected}，须由销售方重开"),
    "R009": (KIND_REISSUE, "该票种（定额发票等）制度明文不接受，须取得增值税发票后重报"),
    "R012": (KIND_MANUAL, "与同日同销售方发票连号，需说明各张发票对应的实际业务内容（此处只提示，不作定性）"),
    "R014": (KIND_MANUAL, "本部门不在预算表内（{expected}），须先落实预算归属或走特批"),
    "R015": (KIND_MANUAL, "票面含针对审核系统的指令性文字，须人工核实票面真实性"),
    "R019": (KIND_SUPPLEMENT, "境外票据须补充汇率与折算人民币金额的说明，由人工核验"),
}


def suggest_advice(findings: list[AuditFinding], policy=None) -> list[dict]:
    """按规则命中项生成补正建议。**只读**，不改变任何状态。

    :param policy: 制度包（可选）。给了就从里面取餐饮人均限额，
                   不给则用制度默认值 150 元 —— 限额只存一处，不在这里另立标准。
    """
    meal_limit = Decimal("150")
    if policy is not None:
        try:
            meal_limit = parse_money(policy.limits.get("meal_per_person", 150))
        except Exception:  # noqa: BLE001
            pass
    items: list[dict] = []
    for f in sorted(findings, key=lambda x: (_SEVERITY_ORDER.get(x.severity, 9), x.rule_id)):
        if f.severity is Severity.PASS:
            continue
        builder = BUILDERS.get(f.rule_id)
        if builder is not None:
            items.append(builder(f, meal_limit) if f.rule_id == "R006" else builder(f))
        elif f.rule_id in FIXED_KIND:
            kind, template = FIXED_KIND[f.rule_id]
            expected = f.evidence.get("expected")
            if isinstance(expected, list):
                expected = "、".join(map(str, expected))
            items.append(_advice(f.rule_id, f.clause, kind, template.format(expected=expected or "")))
        else:
            items.append(_default(f))
    return items


#: 会触发"补情况说明"的规则：金额/时限类超标 —— 说明能救、也能走特批。
EXPLAINABLE_RULES = {"R003", "R005", "R006", "R007"}

#: 处置选项的 code（稳定标识，前端与 ERP 可据此做路由；文案可以改，code 不要改）。
DISP_REVISE = "revise"
DISP_SUPPLEMENT = "supplement"
DISP_EXPLAIN = "explain"
DISP_ESCALATE = "escalate"


def suggest_dispositions(
    findings: list[AuditFinding],
    *,
    applicant: str = "",
    expense_type: str = "",
    issue_date: str = "",
) -> list[dict]:
    """处置选项：把"驳回"变成几条可执行的路径。**只读、确定性、不改变任何状态。**

    系统只**列出**选项，不替人选、不自动执行；选哪条、批不批仍然由人决定。
    每个选项带一句能直接用的文字（模板），空着的地方留给人填 —— 不编造事实，
    也不给"拆单/改费用类型绕过去"这类规避建议。

    - revise     ：退回修改后重新提交（原单不动，生成新单）；
    - supplement ：补齐材料（制度要求"补充材料"的规则）；
    - explain    ：补一份情况说明走特批（限额/时限类超标）；
    - escalate   ：转上级书面特批（存在事实性驳回项时）。
    """
    blocking = [f for f in findings if f.severity is Severity.FAIL]
    nonpass = [f for f in findings if f.severity is not Severity.PASS]
    if not nonpass:
        return []

    advice = suggest_advice(findings)
    kinds = {item["kind"] for item in advice}
    rules = {f.rule_id for f in nonpass}

    out: list[dict] = [
        {
            "code": DISP_REVISE,
            "title": "退回修改",
            "detail": "按补正建议修改申请单后重新提交；原单保留、不改写，生成新单。",
            "requires": "申请人",
        }
    ]
    if kinds & {KIND_SUPPLEMENT, KIND_FIX, KIND_REISSUE}:
        out.append(
            {
                "code": DISP_SUPPLEMENT,
                "title": "补齐材料",
                "detail": "；".join(
                    item["text"]
                    for item in advice
                    if item["kind"] in {KIND_SUPPLEMENT, KIND_FIX, KIND_REISSUE}
                ),
                "requires": "申请人",
            }
        )
    if rules & EXPLAINABLE_RULES:
        who = applicant or "本人"
        when = issue_date or "____"
        what = expense_type or "____"
        out.append(
            {
                "code": DISP_EXPLAIN,
                "title": "补情况说明（走特批）",
                "detail": (
                    f"情况说明模板：{who}于 {when} 因 ________ 发生{what}，"
                    "超出公司标准部分申请特批（真实原因由本人填写，系统不代写）。"
                ),
                "requires": "申请人 + 审批人",
            }
        )
    if blocking:
        out.append(
            {
                "code": DISP_ESCALATE,
                "title": "转上级特批",
                "detail": "由部门负责人与财务负责人书面确认后按特批流程处理；系统只登记选项，不代替审批。",
                "requires": "部门负责人 + 财务负责人",
            }
        )
    return out
