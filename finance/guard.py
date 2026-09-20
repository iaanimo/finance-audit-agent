"""LLM 输出护栏
===============

审核意见那段中文叙述，是这个项目里唯一由模型生成的**给人看的文字**。
它很容易被写成"模型说超标了"——但那就把判定权又还给了模型。

护栏做两件事，缺一不可：

1. **数字必须可溯源** —— 叙述里出现的每一个数字，都必须在 findings 里找得到。
   找不到，就说明模型在编事实。
2. **结论倾向必须一致** —— 模型不能在系统判定"驳回"的单子上写"建议通过"。
   只查数字的话，数字全对但结论说反的叙述照样会放行。

任何一条不过，整段丢弃，换成 :func:`finance.rules.summarize_findings`
生成的模板文字。

这是"受控"的第四条不变量。它同时支撑了演示里的一步：

    把模型关掉，流程照走，只是叙述从 llm 变成 template。

**模型可以润色，不能造事实、不能改变结论，也不能让流程停摆。**

实现上的难点是误杀
------------------
数字那关：**条款号和规则号不是"事实数字"**，规则条数和通过数虽然来自 findings
但不是逐字出现的，所以放行集合要包含 findings 里所有数字、条款号、以及各类计数。

结论那关：只匹配**明确的建议措辞**（"建议通过""建议驳回"），不去匹配制度原文里
的"不得报销"这类词 —— 那是在引用条款，不是在下结论。
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .models import AuditFinding, NarrativeSource, Severity
from .rules import aggregate

# 抓叙述里的数字（含千分位与小数）
_RE_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# 规则号 R001 / R012 —— 先整体剔除，避免把 "001" 当成事实数字
_RE_RULE_ID = re.compile(r"\bR\d{3,}\b")

# 结论倾向。只认**明确的建议措辞**，不认制度原文里的"不得报销"这类引用。
_RE_VERDICT_APPROVE = re.compile(r"建议\s*(予以)?\s*(通过|批准|准予|放行)")
_RE_VERDICT_REJECT = re.compile(r"建议\s*(予以)?\s*(驳回|拒绝|退回)|建议\s*不予通过")


def stated_verdict(narrative: str) -> str | None:
    """从叙述里读模型**自己下的结论**。没下结论返回 None。

    返回 ``"APPROVED"`` / ``"REJECTED"`` / ``None``。
    """
    if not narrative:
        return None
    has_approve = bool(_RE_VERDICT_APPROVE.search(narrative))
    has_reject = bool(_RE_VERDICT_REJECT.search(narrative))
    if has_approve and has_reject:
        return "CONFLICTING"      # 自相矛盾，同样不该放行
    if has_approve:
        return "APPROVED"
    if has_reject:
        return "REJECTED"
    return None


def guard_narrative(
    narrative: str,
    findings: list[AuditFinding],
    fallback: str,
) -> tuple[str, NarrativeSource]:
    """校验模型叙述。通过则原样返回，不通过则返回模板兜底。

    两道关：数字可溯源、结论倾向与系统建议一致。

    :return: ``(最终叙述, 来源标记)``
    """
    if not narrative or not narrative.strip():
        return fallback, NarrativeSource.TEMPLATE

    # 第一关：数字不能编
    allowed = allowed_numbers(findings)
    offending = [
        token for token in _extract_numbers(narrative) if _normalize(token) not in allowed
    ]
    if offending:
        return fallback, NarrativeSource.TEMPLATE

    # 第二关：结论不能反。
    # 官方建议是 PENDING 时，模型也不该替系统下结论 —— 那也是越权。
    stated = stated_verdict(narrative)
    if stated is not None and stated != aggregate(findings).value:
        return fallback, NarrativeSource.TEMPLATE

    return narrative.strip(), NarrativeSource.LLM


def allowed_numbers(findings: list[AuditFinding]) -> set[str]:
    """收集"模型可以引用"的数字集合。

    包含：每条 finding 的证据与结论里的全部数字、条款号、以及汇总计数。
    """
    allowed: set[str] = set()

    for f in findings:
        allowed |= _numbers_in(f.message)
        allowed |= _numbers_in(f.clause)
        allowed |= _numbers_in(f.clause_text)
        for value in _flatten(f.evidence):
            allowed |= _numbers_in(value)

    # 汇总计数：条数本身是事实，允许引用
    counts = {
        len(findings),
        sum(1 for f in findings if f.severity is Severity.PASS),
        sum(1 for f in findings if f.severity is Severity.WARN),
        sum(1 for f in findings if f.severity is Severity.FAIL),
    }
    for c in counts:
        allowed.add(_normalize(str(c)))

    return allowed


def unverifiable_numbers(narrative: str, findings: list[AuditFinding]) -> list[str]:
    """返回叙述里无法溯源到 findings 的数字。**供测试与排查用。**"""
    allowed = allowed_numbers(findings)
    return [t for t in _extract_numbers(narrative) if _normalize(t) not in allowed]


# --------------------------------------------------------------------------
# 内部
# --------------------------------------------------------------------------


def _extract_numbers(text: str) -> list[str]:
    """抓出文本里的数字 token。规则号先剔除，小整数放行交给调用方判断。"""
    if not text:
        return []
    cleaned = _RE_RULE_ID.sub(" ", str(text))
    tokens = []
    for m in _RE_NUMBER.finditer(cleaned):
        token = m.group(0)
        if _normalize(token) in {"", "0"}:
            continue
        tokens.append(token)
    return tokens


def _numbers_in(value) -> set[str]:
    """从任意值里抽出数字的归一化形式。"""
    if value is None:
        return set()
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float, Decimal)):
        return {_normalize(str(value))}
    if isinstance(value, str):
        return {_normalize(t) for t in _extract_numbers(value)}
    return set()


def _flatten(node) -> list:
    """把嵌套的 evidence 结构拍平成一维值列表。"""
    if isinstance(node, dict):
        out = []
        for v in node.values():
            out.extend(_flatten(v))
        return out
    if isinstance(node, (list, tuple, set)):
        out = []
        for v in node:
            out.extend(_flatten(v))
        return out
    return [node]


def _normalize(token: str) -> str:
    """把数字归一成两位小数的字符串，让 600 / 600.0 / 600.00 / 1,650.00 可比。

    注意小额整数也走这条路径：``3`` 归一成 ``3.00``，
    这样"住宿 3 晚"和 evidence 里的 ``nights: 3`` 能对上。
    """
    raw = str(token).replace(",", "").strip()
    if not raw:
        return ""
    try:
        d = Decimal(raw)
    except (InvalidOperation, ValueError):
        return raw
    return f"{d.quantize(Decimal('0.01'))}"
