"""记账凭证草稿生成
==================

审批通过后生成借贷分录。**草稿**的意思很实在：它必须经人工确认才能入账
（制度 6.1）。本模块只负责把分录算对、配平，不负责决定它能不能入账。

本模块是纯函数，零 LLM、零状态 —— 科目来自 ``policies/accounts.yaml`` 的查表，
金额来自发票价税合计。**算钱交给代码**，这是全包贯彻到底的一条线。
"""

from __future__ import annotations

from .models import (
    Invoice,
    ReimbursementRequest,
    Voucher,
    VoucherLine,
    money_str,
    parse_money,
)
from .policy import PolicyBundle
from .rules import RuleContext, _resolved_expense_type


class VoucherError(Exception):
    """无法生成凭证（通常是科目无法归类）。由调用方决定降级策略。"""


def input_tax_deductible(invoice: Invoice, request: ReimbursementRequest, policy: PolicyBundle) -> bool:
    """这张票的进项税额**允许抵扣**吗？三个条件缺一不可：

    1. **专用发票** —— 普通发票的进项税额不可抵扣，税额随价税合计全额进费用；
    2. **本公司可抵扣** —— 一般纳税人（``rules.yaml`` 的
       ``company.input_tax_deductible``）。小规模纳税人取得的发票同样不得抵扣；
    3. **费用用途可抵扣** —— 财税〔2016〕36号 附件1 第二十七条：购进的
       **餐饮服务**、居民日常服务、娱乐服务的进项税额**不得抵扣**。
       餐饮费取得专用发票也不得拆「进项税额」—— 严禁级的会计口径，
       由科目表逐项声明（``accounts.yaml`` 的 ``input_tax_deductible``）。

    还有第四条（不属于集体福利、个人消费等）票面与申请单都看不出来，
    由人工复核 —— 系统只判前三条。查不到费用类型时**保守返回 False**：
    宁可少抵，不可错抵。
    """
    if not policy.input_tax_deductible:
        return False
    if "专用" not in (invoice.invoice_type or ""):
        return False
    ctx = RuleContext(invoice=invoice, request=request, policy=policy)
    expense_type, _ = _resolved_expense_type(ctx)
    spec = next((s for s in policy.accounts if s.expense_type == expense_type), None)
    return bool(spec.input_tax_deductible) if spec else False


# 部门前缀替换：accounts.yaml 里科目默认挂"管理费用"，销售部应改挂"销售费用"
_ACCOUNT_PREFIXES = ("管理费用", "销售费用", "财务费用", "制造费用")


def resolve_debit_account(policy: PolicyBundle, ctx: RuleContext) -> str:
    """查出本单应借记的会计科目。

    两步：① 按费用类型查表拿到基础科目；② 按申请人所属部门替换费用大类前缀。
    """
    expense_type, _ = _resolved_expense_type(ctx)
    if not expense_type:
        raise VoucherError(
            "费用类型无法归类到会计科目表，无法生成凭证（见规则 R011）"
        )

    spec = next((s for s in policy.accounts if s.expense_type == expense_type), None)
    if spec is None:
        raise VoucherError(f"科目表中没有费用类型「{expense_type}」")

    base = spec.debit_account
    prefix = policy.account_prefix_for(ctx.request.department)
    for old in _ACCOUNT_PREFIXES:
        if base.startswith(old) and prefix and prefix != old:
            return prefix + base[len(old):]
    return base


def build_voucher(
    invoice: Invoice,
    request: ReimbursementRequest,
    policy: PolicyBundle,
) -> Voucher:
    """生成一张记账凭证草稿。

    **借方什么时候拆两行（不含税金额 + 进项税额）—— 会计规则，不是排版偏好：**

        借  管理费用-差旅费-住宿费      1,556.60    ← 不含税
        借  应交税费-应交增值税-进项税额    93.40    ← 税额（仅专用发票可抵扣）
        贷  其他应付款-员工报销          1,650.00

    进项税额要**同时满足三个条件**才允许单独成行：

    1. 票是**增值税专用发票**（普通发票的进项税额不可抵扣 —— 税额必须随
       价税合计**全额计入成本费用**，拆出去记「进项税额」是会计错误）；
    2. 本公司**可抵扣**（一般纳税人）。小规模纳税人取得的发票同样不得抵扣，
       由 ``rules.yaml`` 的 ``company.input_tax_deductible`` 控制；
    3. **费用用途可抵扣** —— 餐饮服务等法定不得抵扣的费用（财税〔2016〕36号
       附件1 第二十七条），专票也不拆，由 ``accounts.yaml`` 逐项声明。

    三个条件任一不满足 -> 单行写法：借费用（价税合计）/ 贷往来。
    票面缺「不含税金额」或「税额」时同样退回单行 —— 信息不足以拆。

    **不校验 net + tax == total** —— 那是 R013 的活，而且 R013 现在直接对
    票面三要素做加法校验（见 finance/rules.py::check_voucher_balance），
    不再依赖凭证的行数形状。这里仍**刻意不做「合不上就退回单行」的兜底**：
    不含税金额加税额对不上价税合计，本身就是票面有问题（增值税发票上这
    三者本就必须严丝合缝），凭证如实照抄，由 R013 判 FAIL 报出来。加兜底
    把它抹平，等于替票面圆谎 —— 而"票面自相矛盾"恰恰是最该被抓住的那类问题。

    :raises VoucherError: 科目无法归类，或金额缺失/为零
    """
    ctx = RuleContext(invoice=invoice, request=request, policy=policy)
    debit_account = resolve_debit_account(policy, ctx)
    credit_account = policy.credit_account or "其他应付款-员工报销"

    total = invoice.total if invoice.total is not None else request.amount
    amount = parse_money(total)
    if amount <= 0:
        raise VoucherError(f"凭证金额必须大于零，当前为 {money_str(amount)}")

    lines = [
        VoucherLine(direction="借", account=debit_account, amount=amount),
    ]

    net = invoice.amount
    tax = invoice.tax_amount
    tax_account = policy.input_tax_account
    # 拆行四前提：① 专用发票 ② 本公司可抵扣 ③ 用途可抵扣（餐饮等法定排除）
    # ④ 票面三项齐全且税额 > 0。
    # **不校验 net + tax == total** —— 那是 R013 的活。
    # 这里多写一句"合不上就退回单行"，R013 就永远看不到不合的情况了。
    if (
        input_tax_deductible(invoice, request, policy)
        and net is not None
        and tax is not None
        and invoice.total is not None
        and tax_account
        and parse_money(tax) > 0
    ):
        lines = [
            VoucherLine(direction="借", account=debit_account, amount=parse_money(net)),
            VoucherLine(direction="借", account=tax_account, amount=parse_money(tax)),
        ]

    lines.append(
        VoucherLine(direction="贷", account=credit_account, amount=amount)
    )

    return Voucher(lines=lines, summary=_make_summary(request, invoice))


def _make_summary(request: ReimbursementRequest, invoice: Invoice) -> str:
    """凭证摘要。财务习惯：报销人 + 事项。太长会被财务手工截断，控制在一行内。"""
    parts = []
    if request.applicant:
        parts.append(request.applicant)
    if request.expense_type:
        parts.append(f"{request.expense_type}报销")
    if request.reason:
        parts.append(request.reason)
    if not parts:
        parts.append(f"{invoice.item_name or '费用'}报销")
    return " ".join(parts)


def _display_width(s: str) -> int:
    """字符串在等宽终端里占几列。中文/全角算 2 列。

    Python 的 ``:<24`` 按**字符数**补空格，而中日韩字符显示宽度是 2 ——
    科目名一长（如「应交税费-应交增值税（进项税额）」）整列就歪了。
    """
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def format_voucher(voucher: Voucher) -> str:
    """把凭证渲染成等宽文本，便于终端/日志/演示展示。"""
    account_col = 34
    width = max([_display_width(l.account) for l in voucher.lines] + [0])
    account_col = max(account_col, width + 2)

    lines = ["记账凭证（草稿）", f"摘要：{voucher.summary}", "-" * 52]
    for line in voucher.lines:
        pad = " " * max(1, account_col - _display_width(line.account))
        lines.append(f"  {line.direction}  {line.account}{pad}{money_str(line.amount):>12}")
    lines.append("-" * 52)
    lines.append(
        f"  借方合计 {money_str(voucher.debit_total)}    "
        f"贷方合计 {money_str(voucher.credit_total)}    "
        f"{'平衡' if voucher.balanced else '不平衡'}"
    )
    return "\n".join(lines)
