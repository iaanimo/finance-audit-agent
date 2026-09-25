"""财务报销审核受控 Agent —— 领域包
=====================================

一句话概括本包的设计：**LLM 只管"读"，规则引擎管"判"。**

分层
----
- :mod:`finance.policy`    —— 制度加载（YAML 配置，不是提示词）
- :mod:`finance.rules`     —— 19 条规则，**零 LLM**，判定权的唯一所在
- :mod:`finance.extractor` —— 从发票文件抽取字段（允许 LLM 参与，允许出错）
- :mod:`finance.voucher`   —— 记账凭证草稿
- :mod:`finance.guard`     —— LLM 输出护栏（叙述里的数字必须来自 findings）
- :mod:`finance.audit`     —— 编排，**唯一的状态变更入口**
- :mod:`finance.store`     —— 审核单落盘与审计日志（只追加）

对外只暴露下面这些名字；子模块内部实现可以变。
"""

from .models import (
    AuditFinding,
    AuditResult,
    AuditState,
    Decision,
    Invoice,
    NarrativeSource,
    ReimbursementRequest,
    Severity,
    SuggestedStatus,
    Voucher,
    VoucherLine,
    money_eq,
    money_le,
    money_str,
    parse_chinese_amount,
    parse_money,
)
from .policy import PolicyBundle, PolicyError, load_policy_bundle
from .rules import (
    CheckOutcome,
    HistoryHit,
    HistoryView,
    RuleContext,
    aggregate,
    evaluate,
    summarize_findings,
)

__all__ = [
    # models
    "Invoice",
    "ReimbursementRequest",
    "AuditFinding",
    "AuditResult",
    "AuditState",
    "SuggestedStatus",
    "Severity",
    "Decision",
    "NarrativeSource",
    "Voucher",
    "VoucherLine",
    "parse_money",
    "parse_chinese_amount",
    "money_eq",
    "money_le",
    "money_str",
    # policy
    "PolicyBundle",
    "PolicyError",
    "load_policy_bundle",
    # rules
    "evaluate",
    "aggregate",
    "summarize_findings",
    "RuleContext",
    "CheckOutcome",
    "HistoryView",
    "HistoryHit",
]
