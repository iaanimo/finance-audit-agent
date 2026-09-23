"""数据契约（Data Contracts）
==========================

本模块定义报销审核全链路的数据结构。是整个 finance 包的地基，别的模块只依赖它。

设计要点
--------
1. **金额一律用 Decimal**。浮点数比较是财务系统的经典事故源：
   ``0.1 + 0.2 != 0.3``，金额相等判断若用 ``==`` 迟早出错。
   本模块提供 :func:`parse_money` / :func:`money_eq` / :func:`money_le` 三个助手，
   全包内比金额一律走它们。
2. **每种结论都带证据**。:class:`AuditFinding` 强制携带 ``evidence`` 字典，
   记录"哪个字段、实际值多少、期望值多少"。只看结论无法反驳，看证据才能。
3. **状态机是显式的**。:class:`AuditState` 的六态不是装饰，它对应审核流程里
   真实发生的阶段推进——见 finance/audit.py。
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# 金额助手
# --------------------------------------------------------------------------

CENT = Decimal("0.01")


def parse_money(value: Any) -> Decimal:
    """把任意输入解析成「分」为最小单位的 Decimal。

    容忍发票文本里的常见噪音：千分位逗号、人民币符号、空白。
    解析失败抛 ValueError —— 调用方必须显式处理，不允许静默当作 0。
    """
    if isinstance(value, Decimal):
        d = value
    elif isinstance(value, (int, float)):
        d = Decimal(str(value))          # 先转 str，避免二进制浮点误差被带进来
    else:
        cleaned = str(value)
        for noise in ("￥", "¥", ",", " ", " ", "元"):
            cleaned = cleaned.replace(noise, "")
        cleaned = cleaned.strip()
        if not cleaned:
            raise ValueError("金额为空")
        try:
            d = Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(f"无法解析金额: {value!r}") from exc
    return d.quantize(CENT, rounding=ROUND_HALF_UP)


def money_eq(a: Any, b: Any) -> bool:
    """金额**相等**判断，容忍 1 分的舍入残差。

    容差在这里是必要的：人均金额、每晚单价这类派生值做除法后要
    ``quantize`` 到分，1 分的残差是分摊的必然产物，不是差异。

    注意不要用 ``==`` 直接比 Decimal —— 那会把合法的舍入残差判成不等。
    """
    return abs(parse_money(a) - parse_money(b)) <= CENT


def money_le(a: Any, b: Any) -> bool:
    """``a <= b`` 的金额版本 —— **精确比较，不带容差**。

    这里刻意与 :func:`money_eq` 不同。限额判断（"住宿不超过 600 元/晚"）
    一旦带上 1 分容差，限额实际就变成了 600.01，超限 1 分的发票会被放行。
    而且全程用 Decimal 本来就没有二进制浮点的表示误差，容差是多余的。

    （这个 bug 是被 tests/test_finance.py 的
    ``test_money_uses_decimal_not_float`` 抓出来的。）
    """
    return parse_money(a) <= parse_money(b)


def money_str(value: Any) -> str:
    """金额的展示形式，如 ``1,650.00``。"""
    return f"{parse_money(value):,.2f}"


# --------------------------------------------------------------------------
# 中文大写金额
# --------------------------------------------------------------------------
#
# 用于 R016 大小写一致性校验。为什么这条规则值得单独写一个解析器：
# 大小写不符是**票面被篡改的典型特征**（改小写不改大写，或反之），
# 正规财务审单一定会核这一项。它是"一眼看出你真做过财务"的规则。

_CN_DIGITS = {
    "零": 0, "〇": 0,
    "壹": 1, "贰": 2, "叁": 3, "肆": 4, "伍": 5, "陆": 6, "柒": 7, "捌": 8, "玖": 9,
    # 容忍小写写法（有些票据或手写件会用）
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "两": 2,
}
_CN_UNITS = {"拾": 10, "佰": 100, "仟": 1000, "十": 10, "百": 100, "千": 1000}
_CN_SECTIONS = {"万": 10**4, "亿": 10**8}

_CN_AMOUNT_CHARS = set(_CN_DIGITS) | set(_CN_UNITS) | set(_CN_SECTIONS) | set("圆元角分整正")


def parse_chinese_amount(text: Any) -> Decimal | None:
    """把「壹佰叁拾壹圆柒角叁分」解析成 ``Decimal('131.73')``。

    无法识别时返回 ``None``（**不抛异常**）—— 调用方据此判定
    "大写栏无法核对"，而不是让整条规则崩掉。
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    s = s.replace("正", "整").replace(" ", "").replace("　", "")
    s = s.rstrip("整")
    if not s or not set(s) <= _CN_AMOUNT_CHARS:
        return None

    # 切开「圆/元」：前面是整数部分，后面是角分
    int_part, frac_part = s, ""
    for yuan in ("圆", "元"):
        if yuan in s:
            int_part, _, frac_part = s.partition(yuan)
            break
    else:
        # 整串没有「圆/元」。若含角/分，那整串都是小数部分（如「柒角」「零角叁分」）；
        # 否则整串就是整数（如「壹拾伍」）。
        if "角" in s or "分" in s:
            int_part, frac_part = "", s

    integer = _cn_integer(int_part)
    if integer is None:
        return None
    fraction = _cn_fraction(frac_part)
    if fraction is None:
        return None
    return (Decimal(integer) + fraction).quantize(CENT, rounding=ROUND_HALF_UP)


def _cn_integer(s: str) -> int | None:
    """按「节」解析中文整数：万、亿 分段累加。"""
    if not s:
        return 0
    total = 0
    section = 0
    number = 0
    for ch in s:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            # "拾" 前无数字表示 1，如「拾伍」= 15
            section += (number if number else 1) * _CN_UNITS[ch]
            number = 0
        elif ch in _CN_SECTIONS:
            section += number
            total += section * _CN_SECTIONS[ch]
            section = 0
            number = 0
        else:
            return None
    return total + section + number


def _cn_fraction(s: str) -> Decimal | None:
    """解析「柒角叁分」这类小数部分。"""
    if not s:
        return Decimal("0.00")
    result = Decimal("0")
    if "角" in s:
        head = s.split("角", 1)[0]
        digit = _last_digit(head)
        if digit is None:
            return None
        result += Decimal(digit) / 10
        s = s.split("角", 1)[1]
    if "分" in s:
        head = s.split("分", 1)[0]
        digit = _last_digit(head)
        if digit is None:
            return None
        result += Decimal(digit) / 100
        s = s.split("分", 1)[1]
    if s:
        return None
    return result.quantize(CENT, rounding=ROUND_HALF_UP)


def _last_digit(s: str) -> int | None:
    """取「零」「叁」这类单字数字；空串视为 0。"""
    if not s:
        return 0
    for ch in reversed(s):
        if ch in _CN_DIGITS:
            return _CN_DIGITS[ch]
    return None


# --------------------------------------------------------------------------
# 枚举
# --------------------------------------------------------------------------


class Severity(str, Enum):
    """单条规则的判定结果。

    三态语义（这是本项目的立论之一）：

    - ``PASS`` 通过
    - ``FAIL`` **事实性违规** —— 制度原文写"不得报销""不予受理"，可复现的算术结论
    - ``WARN`` **概率性怀疑** —— 制度原文写"提交人工复核""退回补充"，
      系统只提示，不替人定罪
    """

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


class AuditState(str, Enum):
    """审核单状态。推进顺序见 finance/audit.py::run_audit。

    抽取 -> 校验 -> 查重 -> 预算 -> 草稿 -> 待人工复核 -> 已批准/已驳回
    """

    EXTRACTED = "extracted"
    VALIDATED = "validated"
    DUPLICATE_CHECKED = "duplicate_checked"
    BUDGET_CHECKED = "budget_checked"
    DRAFT_CREATED = "draft_created"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class SuggestedStatus(str, Enum):
    """系统**建议**的结论。注意是建议，不是决定 —— 最终决定权在人。"""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


class Decision(str, Enum):
    """人工做出的最终决定。"""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class NarrativeSource(str, Enum):
    """审核意见叙述的来源，用于演示"模型挂了流程照走"。"""

    LLM = "llm"
    TEMPLATE = "template"


# --------------------------------------------------------------------------
# 发票
# --------------------------------------------------------------------------


class Invoice(BaseModel):
    """从发票中抽取出的结构化字段。

    字段命名对齐中国增值税发票票面用语。抽取允许出错 —— 下一层的规则引擎
    会逐条校验，错了会被抓住，这正是分层的目的。
    """

    invoice_code: str = Field(default="", description="发票代码（数电票可能为空）")
    invoice_number: str = Field(default="", description="发票号码")
    invoice_type: str = Field(default="", description="发票类型，如 电子发票（普通发票）")
    issue_date: date | None = Field(default=None, description="开票日期")

    buyer_name: str = Field(default="", description="购买方名称")
    buyer_tax_id: str = Field(default="", description="购买方纳税人识别号")
    seller_name: str = Field(default="", description="销售方名称")
    seller_tax_id: str = Field(default="", description="销售方纳税人识别号")

    item_name: str = Field(default="", description="项目名称（货物或服务）")
    amount: Decimal | None = Field(default=None, description="金额（不含税）")
    tax_rate: str = Field(default="", description="税率，如 3%")
    tax_amount: Decimal | None = Field(default=None, description="税额")
    total: Decimal | None = Field(default=None, description="价税合计（小写）")
    total_in_words: str = Field(default="", description="价税合计（大写）")

    remark: str = Field(default="", description="备注栏")

    raw_text: str = Field(default="", description="抽取来源的原始文本，证据链的根")
    source_file: str = Field(default="", description="来源文件名")
    extraction_method: str = Field(
        default="", description="抽取方式：pdf_text / vision —— 演示时要能说清数据从哪来"
    )

    @field_validator("amount", "tax_amount", "total", mode="before")
    @classmethod
    def _coerce_money(cls, v: Any) -> Any:
        """把抽取到的金额统一成 Decimal。空值保留 None（由规则引擎判定为缺失）。"""
        if v is None or v == "":
            return None
        return parse_money(v)

    def key(self) -> str:
        """发票唯一键：代码 + 号码。查重（R004）用的就是它。"""
        return f"{self.invoice_code}-{self.invoice_number}"


# --------------------------------------------------------------------------
# 报销申请
# --------------------------------------------------------------------------


class ReimbursementRequest(BaseModel):
    """报销申请单。发票之外的信息由申请人填写。"""

    applicant: str = Field(default="", description="申请人")
    department: str = Field(default="", description="所属部门，决定预算与科目前缀")
    expense_type: str = Field(default="", description="费用类型，如 市内交通费")
    amount: Decimal = Field(description="申请报销金额")
    reason: str = Field(default="", description="事由")
    submit_date: date = Field(description="提交日期")

    # 以下为按费用类型选填的附加信息
    city: str = Field(default="", description="住宿/差旅目的地城市")
    nights: int | None = Field(default=None, description="住宿晚数")
    headcount: int | None = Field(default=None, description="用餐人数")

    has_itemized_list: bool = Field(default=False, description="是否附采购清单")

    # ---- 申报扩展信息（人工填写/确认）----
    # 分摊拆凭证是 Phase 2：当前透传 + 留痕，不参与借贷拆分（README 注明）。
    project: str = Field(default="", description="归属项目")
    cost_center: str = Field(default="", description="成本中心")
    allocation_ratio: str = Field(default="", description="分摊比例，如 50%")
    note: str = Field(default="", description="备注")

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, v: Any) -> Any:
        return parse_money(v)


# --------------------------------------------------------------------------
# 审核结论
# --------------------------------------------------------------------------


class AuditFinding(BaseModel):
    """一条规则的判定结果 —— 审核结论的最小单元。

    ``evidence`` 是重点：它让复核人能验算系统的判断，而不是选择相信它。
    """

    rule_id: str
    clause: str = Field(description="制度条款号，如 4.2")
    title: str
    clause_text: str = Field(default="", description="制度原文")
    severity: Severity
    message: str = Field(description="人话结论")
    evidence: dict[str, Any] = Field(default_factory=dict, description="证据字段")
    error: str | None = Field(
        default=None, description="规则执行异常时的错误信息（异常降级为 WARN）"
    )

    @property
    def is_blocking(self) -> bool:
        return self.severity is Severity.FAIL


class VoucherLine(BaseModel):
    """凭证的一行分录。"""

    direction: Literal["借", "贷"]
    account: str
    amount: Decimal

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, v: Any) -> Any:
        return parse_money(v)


class Voucher(BaseModel):
    """记账凭证草稿。必须人工确认后才能入账（制度 6.1）。"""

    lines: list[VoucherLine]
    summary: str = Field(default="", description="摘要")

    @property
    def debit_total(self) -> Decimal:
        return sum((l.amount for l in self.lines if l.direction == "借"), Decimal("0.00"))

    @property
    def credit_total(self) -> Decimal:
        return sum((l.amount for l in self.lines if l.direction == "贷"), Decimal("0.00"))

    @property
    def balanced(self) -> bool:
        """借贷是否平衡 —— **精确相等，不带容差**。

        「有借必有贷、借贷必相等」是借贷记账法里的绝对等式，会计上不存在
        "差一分也算平"的凭证。``money_eq`` 的 1 分容差是给「人均」「每晚」
        这类**除法派生值**准备的（制度 4.2、4.3），用在这里会把
        「票面不含税金额 + 税额 ≠ 价税合计」这种自相矛盾悄悄抹平。
        """
        return self.debit_total == self.credit_total


class AuditResult(BaseModel):
    """一次审核的完整结果。这是落盘和展示的顶层对象。"""

    audit_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: AuditState = AuditState.EXTRACTED

    invoice: Invoice
    request: ReimbursementRequest
    findings: list[AuditFinding] = Field(default_factory=list)

    suggested_status: SuggestedStatus = SuggestedStatus.PENDING
    narrative: str = Field(default="", description="审核意见（给人看的叙述）")
    narrative_source: NarrativeSource = NarrativeSource.TEMPLATE

    voucher: Voucher | None = None

    # 人工决定的留痕（制度 2.3 / 6.2）
    decision: Decision | None = None
    operator: str = ""
    override_reason: str = Field(
        default="", description="推翻系统建议时的书面理由 —— 必填，否则服务端 400"
    )
    decided_at: datetime | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ---- 派生信息 ----

    @property
    def blocking_findings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity is Severity.FAIL]

    @property
    def warning_findings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity is Severity.WARN]

    def is_overridden_now(self, decision: Decision) -> bool:
        """给定一个待做的决定，判断它是否构成"推翻系统建议"。

        ``PENDING`` 不算系统意见 —— 那是"系统发现疑点、拒绝表态"。
        此时人做任何决定都是**补上系统没给的判断**，不是推翻它。
        只有系统明确建议了 APPROVED / REJECTED 而人被反着来，才算推翻，
        才需要书面理由（制度 2.3）。
        """
        if self.suggested_status is SuggestedStatus.PENDING:
            return False
        return decision.value != self.suggested_status.value

    @property
    def is_overridden(self) -> bool:
        """已做的人工决定是否推翻了系统建议。"""
        if self.decision is None:
            return False
        return self.is_overridden_now(self.decision)

    def narrative_evidence_block(self) -> str:
        """把逐条判定渲染成文本块，供叙述提示词与调试使用。

        刻意带上 evidence —— 让模型看见"实际值/期望值"，它就不需要编数字；
        而一旦它编了，护栏会抓到。
        """
        import json as _json

        lines: list[str] = []
        for f in self.findings:
            lines.append(f"[{f.rule_id}] 制度{f.clause} {f.title} -> {f.severity.value}")
            lines.append(f"    {f.message}")
            if f.evidence:
                lines.append(f"    证据: {_json.dumps(f.evidence, ensure_ascii=False)}")
        return "\n".join(lines)

    def to_json_dict(self) -> dict:
        """转成可 `json.dumps` 的 dict —— Decimal 转 float 并保留两位。

        另外补上几个**派生字段**：``model_dump`` 只输出模型字段，
        而 ``is_overridden`` / 借贷合计是 property，前端需要它们
        （"已推翻"标签、凭证合计行都要用），所以在这里显式附上。
        """
        data = self.model_dump(mode="json")
        _round_money_fields(data)

        data["overridden"] = self.is_overridden
        if data.get("voucher"):
            data["voucher"]["debit_total"] = float(self.voucher.debit_total)
            data["voucher"]["credit_total"] = float(self.voucher.credit_total)
            data["voucher"]["balanced"] = self.voucher.balanced
        return data


def _round_money_fields(node: Any) -> None:
    """递归把 dict/list 里的金额字符串转成两位小数的 float。原地修改。"""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and k in {
                "amount", "tax_amount", "total", "debit_total", "credit_total",
            }:
                try:
                    node[k] = float(parse_money(v))
                    continue
                except (ValueError, InvalidOperation):
                    pass
            _round_money_fields(v)
    elif isinstance(node, list):
        for item in node:
            _round_money_fields(item)
