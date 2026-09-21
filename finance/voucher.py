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

    **借方拆成两行（不含税金额 + 进项税额）**，只要票面写了这三项就照票面拆：

        借  管理费用-差旅费-住宿费      1,556.60    ← 不含税
        借  应交税费-应交增值税-进项税额    93.40    ← 税额
        贷  其他应付款-员工报销          1,650.00

    **为什么必须拆开 —— 这是 R013 能不能成立的前提。**

    如果借方永远只写一行总额、而这个总额又正是贷方那个数，那么「借贷平衡」
    就是一条**恒真式**：无论票面写成什么样都通过，等于没校验。

    拆开之后，借方合计是 `不含税 + 税额` **加出来的**，贷方是票面价税合计 ——
    两者是否相等，取决于**票面自己自不自洽**，不再取决于这段代码怎么写。

    所以这里**刻意不做「合不上就退回单行」的兜底**。不含税金额加税额对不上
    价税合计，本身就是票面有问题（增值税发票上这三者本就必须严丝合缝）；
    凭证如实照抄，由 R013 判 FAIL 报出来。加兜底把它抹平，等于替票面圆谎 ——
    而"票面自相矛盾"恰恰是最该被抓住的那类问题。

    票面缺「不含税金额」或「税额」时退回单行写法：信息不足以判断，
    此时借贷相等是个**事实**，不是恒真式。

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
    # 三项都在才拆。**不校验 net + tax == total** —— 那是 R013 的活。
    # 这里多写一句"合不上就退回单行"，R013 就永远看不到不合的情况了。
    if (
        net is not None
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
