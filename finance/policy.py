"""制度加载器
============

把 ``policies/`` 下的四份配置读成一个 :class:`PolicyBundle`，供规则引擎使用。

为什么制度是**配置**而不是塞进提示词
------------------------------------
把"请遵守公司制度"写进 system prompt，模型判不判、判成什么，你无法回答，
也无法复现。改成配置之后：

- 每条规则绑定一个**条款号**，结论能直接指回制度原文；
- 阈值改一处（rules.yaml）即全局生效，不会与制度脱节；
- ``reimbursement.md`` 的原文被一起读进来，供一致性测试比对（见 tests）。

本模块**不依赖 config.settings**，也**不在 import 时读任何全局状态** ——
政策文件就在包内（``finance/policies/``），路径显式可传，便于测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import Severity

DEFAULT_POLICY_DIR = Path(__file__).resolve().parent / "policies"


# --------------------------------------------------------------------------
# 规格对象
# --------------------------------------------------------------------------


def _require(d: dict[str, Any], key: str, kind: str, ident_key: str = "") -> Any:
    """取制度 YAML 里的必填字段；缺了就报**能定位**的错误。

    直接用 ``d[key]`` 会抛裸 ``KeyError: 'checker'`` —— 不说是哪个文件、
    哪条规则，和 :class:`PolicyError` 的契约（"启动时就该炸，且要知道炸在哪"）
    对不上。少了这个包装，排查的人得自己去翻 YAML 猜。
    """
    if key not in d or d[key] in (None, ""):
        ident = d.get(ident_key) or "?"
        raise PolicyError(f"{kind}「{ident}」缺少必填字段 {key!r}")
    return d[key]


@dataclass(frozen=True)
class RuleSpec:
    """一条规则的元数据。**不含判定逻辑** —— 逻辑在 rules.py 的 checker 函数里。

    这种拆分是刻意的：rules.yaml 说"规则是什么、依据哪条制度"，
    rules.py 说"怎么判"。两者由 rule_id 和 checker 名字绑定，
    一致性由测试保证。
    """

    rule_id: str
    clause: str
    title: str
    clause_text: str
    severity_on_fail: Severity
    checker: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuleSpec":
        # 必填字段按**声明顺序**逐个检查：YAML 改坏时，报出来的第一条就是
        # 从上往下看的第一处缺口，不用来回试。
        rule_id = _require(d, "rule_id", "规则")
        clause = str(_require(d, "clause", "规则", "rule_id"))
        title = _require(d, "title", "规则", "rule_id")
        raw_severity = _require(d, "severity_on_fail", "规则", "rule_id")
        checker = _require(d, "checker", "规则", "rule_id")
        try:
            severity = Severity(raw_severity)
        except ValueError as exc:
            raise PolicyError(
                f"规则「{rule_id}」的 severity_on_fail 取值非法：{raw_severity!r}"
                f"（只能是 {'、'.join(s.value for s in Severity)}）"
            ) from exc
        return cls(
            rule_id=rule_id,
            clause=clause,
            title=title,
            clause_text=d.get("clause_text", ""),
            severity_on_fail=severity,
            checker=checker,
        )


@dataclass(frozen=True)
class AccountSpec:
    """费用类型 -> 会计科目的映射项。"""

    expense_type: str
    keywords: tuple[str, ...]
    debit_account: str
    #: 该类费用取得的专用发票，进项税额**是否允许抵扣**。
    #: 餐饮/居民日常/娱乐服务**法定不得抵扣**（财税〔2016〕36号 附件1 第二十七条），
    #: 在 accounts.yaml 里逐项声明，缺省 True（如办公用品、通讯费）。
    input_tax_deductible: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AccountSpec":
        return cls(
            expense_type=_require(d, "expense_type", "科目映射"),
            keywords=tuple(d.get("keywords", [])),
            debit_account=_require(d, "debit_account", "科目映射", "expense_type"),
            input_tax_deductible=bool(d.get("input_tax_deductible", True)),
        )


@dataclass(frozen=True)
class VatCategory:
    """一类应税行为的适用税率。

    ``rates`` 是**法定税率**的集合。注意判定时的放行集合还要并上
    简易计税征收率 —— 见 :meth:`PolicyBundle.allowed_vat_rates`。
    """

    name: str
    keywords: tuple[str, ...]
    rates: tuple[int, ...]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VatCategory":
        return cls(
            name=_require(d, "name", "税率类别"),
            keywords=tuple(d.get("keywords", [])),
            rates=tuple(int(r) for r in d.get("rates", [])),
        )


@dataclass(frozen=True)
class DepartmentBudget:
    """部门年度预算快照。"""

    name: str
    annual_budget: Any          # Decimal
    used: Any                   # Decimal

    @property
    def remaining(self):
        from .models import parse_money
        return parse_money(self.annual_budget) - parse_money(self.used)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DepartmentBudget":
        from .models import parse_money
        name = _require(d, "name", "部门预算")
        try:
            annual_budget = parse_money(_require(d, "annual_budget", "部门预算", "name"))
            used = parse_money(_require(d, "used", "部门预算", "name"))
        except ValueError as exc:
            raise PolicyError(f"部门预算「{name}」的金额无法解析：{exc}") from exc
        return cls(name=name, annual_budget=annual_budget, used=used)


@dataclass
class PolicyBundle:
    """一次加载得到的完整制度包。规则引擎只认这个对象，不认文件。"""

    company_name: str
    company_tax_id: str
    limits: dict[str, Any]
    accepted_invoice_types: list[str]
    rules: list[RuleSpec]
    accounts: list[AccountSpec]
    credit_account: str
    input_tax_account: str
    department_account_prefix: dict[str, str]
    budgets: dict[str, DepartmentBudget]
    vat_categories: list[VatCategory] = field(default_factory=list)
    simplified_levy_rate: int = 3
    #: 票种大类关键词表（rules.yaml 的 ticket_kinds）—— 规则路由的依据。
    ticket_kinds: dict[str, list] = field(default_factory=dict)
    #: 规则适用票种（rules.yaml 的 rule_scope）。未列出的规则 = 全票种适用。
    rule_scope: dict[str, list] = field(default_factory=dict)
    #: 本公司是否可抵扣进项税额（一般纳税人 = True，小规模纳税人 = False）。
    #: 与「专用发票」两个条件**同时**成立才拆进项税额行 —— 见 finance/voucher.py。
    input_tax_deductible: bool = True
    policy_md: str = ""
    fiscal_year: int = 2026

    # 便于按 id 取规则
    _by_id: dict[str, RuleSpec] = field(default_factory=dict, repr=False)

    def rule(self, rule_id: str) -> RuleSpec | None:
        if not self._by_id:
            self._by_id = {r.rule_id: r for r in self.rules}
        return self._by_id.get(rule_id)

    def rule_ids(self) -> list[str]:
        return [r.rule_id for r in self.rules]

    def ticket_kind(self, invoice_type: str) -> str:
        """票种大类：``vat / transport / voucher / foreign / other / unknown``。

        识别顺序固定：**增值税票系优先**（认票面自称的结构词，如"电子发票/
        增值税/数电票"），再归交通/财政/境外。判据是**票种自称**不是业务内容 ——
        「通行费电子发票」是真增值税票，不能被"通行费"抢走（C2 教训）。
        空类型 = ``unknown`` —— **不跳过任何规则**（缺信息时全量跑）。
        """
        text = str(invoice_type or "")
        if not text.strip():
            return "unknown"
        # 配置缺失 = 无法归类 -> unknown（**不跳过任何规则**）。
        # C4 残留教训：这里曾掉进 keyword 循环后 `return "other"`，
        # 于是删掉配置块 = 所有票种变 other = 增值税规则全 N/A = 静默全面漏查。
        if not self.ticket_kinds:
            return "unknown"
        low = text.lower()
        for kind in ("vat", "transport", "voucher", "foreign"):
            for kw in self.ticket_kinds.get(kind, []):
                if kw and str(kw).lower() in low:
                    return kind
        return "other"

    def rule_in_scope(self, rule_id: str, kind: str) -> bool:
        """这条规则对该票种适用吗？未配置 = 适用；unknown 票种 = 一律适用。"""
        scope = self.rule_scope.get(rule_id)
        if not scope or kind == "unknown":
            return True
        return kind in scope

    def vat_category_for(self, item_name: str) -> VatCategory | None:
        """按票面项目名称匹配应税行为类别。匹配不到返回 None。"""
        if not item_name:
            return None
        text = str(item_name)
        for category in self.vat_categories:
            for kw in category.keywords:
                if kw and kw in text:
                    return category
        return None

    def allowed_vat_rates(self, category: VatCategory) -> set[int]:
        """该类别的**放行税率集合** = 法定税率 ∪ 简易计税征收率。

        为什么要并上征收率：发票票面**看不出销售方是一般纳税人还是小规模纳税人**。
        小规模纳税人适用简易计税，票面税率就是 3%。硬卡法定税率会把
        每一张小规模纳税人的票都误判成"税率错误"—— 宁可漏报，不可误杀。
        """
        return set(category.rates) | {self.simplified_levy_rate}

    def budget_for(self, department: str) -> DepartmentBudget | None:
        return self.budgets.get(department)

    def account_prefix_for(self, department: str) -> str:
        """部门决定科目挂管理费用还是销售费用；未列出的部门一律管理费用。"""
        return self.department_account_prefix.get(department, "管理费用")


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------


class PolicyError(Exception):
    """制度文件缺漏或格式错误。启动时就该炸，不要等到审核中途。"""


def load_policy_bundle(policy_dir: str | Path | None = None) -> PolicyBundle:
    """读取 ``policies/`` 下的全部制度文件。

    :param policy_dir: 制度目录，默认包内的 ``finance/policies``
    :raises PolicyError: 文件缺失或解析失败
    """
    base = Path(policy_dir) if policy_dir else DEFAULT_POLICY_DIR
    if not base.is_dir():
        raise PolicyError(f"制度目录不存在: {base}")

    rules_doc = _read_yaml(base / "rules.yaml")
    accounts_doc = _read_yaml(base / "accounts.yaml")
    budgets_doc = _read_yaml(base / "budgets.yaml")
    policy_md = _read_text(base / "reimbursement.md")
    vat_doc = rules_doc.get("vat_rates", {}) or {}

    company = rules_doc.get("company", {})
    rules = [RuleSpec.from_dict(d) for d in rules_doc.get("rules", [])]
    if not rules:
        raise PolicyError("rules.yaml 里没有任何规则")

    # ---- 票种路由配置的加载校验（C4：配置漂移必须**启动就炸**，不许静默失效）----
    # 项目纪律是"制度文件缺漏/格式错 -> PolicyError"；这两个配置块曾用 dict()
    # 直存，写错 key、整块删掉都静默退回误判状态 —— 财务系统里这等于配置在漂移。
    # C4 残留补丁：两块必须**成对出现**（缺一半=路由状态不明）；两块全缺 =
    # 不启用票种路由（ticket_kind() 返回 unknown，全部规则照跑，宁多勿漏）。
    _raw_kinds = rules_doc.get("ticket_kinds")
    _raw_scope = rules_doc.get("rule_scope")
    if (_raw_kinds is None) != (_raw_scope is None):
        raise PolicyError(
            "rules.yaml 的 ticket_kinds 与 rule_scope 必须成对出现 —— "
            "缺一半则路由状态不明，禁止静默"
        )
    _KNOWN_KINDS = {"vat", "transport", "voucher", "foreign", "other"}
    ticket_kinds = dict(_raw_kinds or {})
    rule_scope = dict(_raw_scope or {})
    for kind, kws in ticket_kinds.items():
        if kind not in _KNOWN_KINDS:
            raise PolicyError(f"rules.yaml ticket_kinds 含未知票种「{kind}」（合法：{sorted(_KNOWN_KINDS)}）")
        if not isinstance(kws, list):
            raise PolicyError(f"rules.yaml ticket_kinds.{kind} 必须是关键词列表")
    _known_rule_ids = {r.rule_id for r in rules}
    for rid, kinds in rule_scope.items():
        if rid not in _known_rule_ids:
            raise PolicyError(f"rules.yaml rule_scope 引用了不存在的规则「{rid}」")
        bad = [k for k in kinds if k not in _KNOWN_KINDS]
        if bad or not isinstance(kinds, list):
            raise PolicyError(f"rules.yaml rule_scope.{rid} 含非法票种 {bad or kinds!r}")

    budgets = {
        spec.name: spec
        for spec in (
            DepartmentBudget.from_dict(d) for d in budgets_doc.get("departments", [])
        )
    }

    return PolicyBundle(
        company_name=company.get("name", ""),
        company_tax_id=company.get("tax_id", ""),
        limits=rules_doc.get("limits", {}),
        accepted_invoice_types=list(rules_doc.get("accepted_invoice_types", [])),
        rules=rules,
        accounts=[AccountSpec.from_dict(d) for d in accounts_doc.get("accounts", [])],
        credit_account=accounts_doc.get("credit_account", ""),
        input_tax_account=accounts_doc.get("input_tax_account", ""),
        department_account_prefix=dict(accounts_doc.get("department_account_prefix", {})),
        budgets=budgets,
        vat_categories=[VatCategory.from_dict(d) for d in vat_doc.get("categories", [])],
        simplified_levy_rate=int(vat_doc.get("simplified_levy_rate", 3)),
        input_tax_deductible=bool(company.get("input_tax_deductible", True)),
        ticket_kinds=ticket_kinds,
        rule_scope=rule_scope,
        policy_md=policy_md,
        fiscal_year=int(budgets_doc.get("fiscal_year", 2026)),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PolicyError(f"制度文件缺失: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyError(f"制度文件解析失败 {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise PolicyError(f"制度文件顶层必须是映射: {path.name}")
    return data


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise PolicyError(f"制度文件缺失: {path}")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 制度原文解析（供一致性测试用）
# --------------------------------------------------------------------------

# 两种标题写法都认：「**3.1 标题**」（号题同粗）与「**2.3** 标题」（只粗号）。
# 曾漏掉后者 —— 2.3 正是"推翻系统建议必须书面说明理由"那条，三方一致测试看不见它。
_CLAUSE_HEADING = re.compile(r"^\*\*(\d+\.\d+)(?:\s+([^*]+))?\*\*", re.MULTILINE)


def parse_clause_headings(policy_md: str) -> dict[str, str]:
    """从 ``reimbursement.md`` 里抽出所有条款号 -> 标题。

    只用于**测试**：验证 rules.yaml 里的每个 clause 都能在制度原文里找到。
    这让"代码里的规则"和"制度里的条款"之间不可能悄悄脱节。
    """
    return {
        m.group(1): (m.group(2) or "").strip()
        for m in _CLAUSE_HEADING.finditer(policy_md)
    }
