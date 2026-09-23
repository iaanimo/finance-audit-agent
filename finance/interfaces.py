"""第三方接入接口（生产接入点）
================================

本项目**不是生产级产品**：所有需要接第三方的能力都在这里**留好接口**，
默认实现一律"不接入但如实注明"，**绝不假装检查过** —— 这与全项目
"信息缺失转人工、能力缺口不冒充通过"是同一条原则。

四个接入点
----------
1. :class:`InvoiceVerifier` —— 发票查验（国家税务总局查验平台 / 航信 / 百望）。
   接入后 R018 按查验结果判定；不接入时 R018 跳过并注明"由人工核验"。
2. :class:`OperatorDirectory` —— 身份与岗位分离（SSO / RBAC）。
   《企业内部控制基本规范》要求制单与审核不得同人 —— 生产**必须**接入；
   本项目默认操作人自由文本（README「合规边界」如实列明这个缺口）。
3. :class:`BudgetProvider` —— 预算系统（ERP / OA）。默认读静态 ``budgets.yaml`` 快照。
4. :class:`SubmissionAnchor` —— 收单锚点（服务器收单时间 / OA 单号）。
   默认提交日期由申请人手填（README「已知限制」第一条）。

已经接过、不需要再留的
----------------------
- 视觉模型：``tools/vision.py``（默认关，``use_vision=True`` 才走）
- 叙述 LLM：``core/llm_factory.py``（可关，模板叙述兜底）
- 查重台账存储：``finance.rules.HistoryView``（默认 JSON 文件扫盘，可换数据库）

接入三原则
----------
- **未接入 = 能力缺口的事实**，写进结论和日志，不隐瞒也不定罪；
- **接入失败**（超时/限流/平台不可用）≠ 违规 —— 转人工，绝不判 FAIL；
- **判定权不外溢**：查验结果只有规则引擎（R018）能判，接口本身无判定权。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable

from .models import Invoice

# --------------------------------------------------------------------------
# 发票查验
# --------------------------------------------------------------------------

#: 查验通过：真伪、状态均正常
VERIFIED = "verified"
#: 查验不通过：查无此票、已作废、已红冲 —— 事实性违规
FAILED = "failed"
#: 查验未能完成：平台超时、限流、**或根本没接第三方** —— 能力缺口，转人工
UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class VerificationResult:
    """一次发票查验的结果。**不含判定** —— 判定是 R018 的事。

    为什么不直接返回 bool：查验世界是三态的（通过 / 不通过 / 没查成），
    压成两态会把"没查成"悄悄变成"通过"或"不通过"，两边都是撒谎。
    """

    status: str = UNAVAILABLE
    reason: str = ""
    provider: str = ""
    checked_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InvoiceVerifier(Protocol):
    """发票查验平台适配器（税局查验平台 / 航信 / 百望等）。"""

    def verify(self, invoice: Invoice) -> VerificationResult:
        """查验一张发票的真伪与状态（正常 / 作废 / 红冲）。

        实现约定：平台超时、限流、网络不通**不要抛异常**，返回
        ``VerificationResult(status=UNAVAILABLE, reason=...)`` ——
        "查不了"是能力缺口，由 R018 转人工；抛异常会把整单审核带崩。
        """


# --------------------------------------------------------------------------
# 身份与岗位分离
# --------------------------------------------------------------------------


@runtime_checkable
class OperatorDirectory(Protocol):
    """操作人目录 + 岗位分离校验（SSO / RBAC 的适配面）。

    《企业内部控制基本规范》：不相容职务分离 —— 制单与审核不得同人。
    生产接入后由 :func:`finance.audit.decide` 调用 ``can_decide`` 把关。
    """

    def can_decide(self, operator: str, result: Any) -> tuple[bool, str]:
        """``(是否允许, 不允许的原因)``。reason 会进审计日志与 4xx 报错。

        至少应校验：操作人身份真实存在、与申请人不同人、具备复核角色。
        """


# --------------------------------------------------------------------------
# 预算
# --------------------------------------------------------------------------


@runtime_checkable
class BudgetProvider(Protocol):
    """预算来源适配面（ERP / OA 的预算模块）。

    默认实现就是读 ``policies/budgets.yaml`` 静态快照；生产应换成
    "年度预算 − 已审批累加"的实时查询。返回对象需有
    ``annual_budget / used / remaining``（对齐 ``policy.DepartmentBudget``）。
    """

    def budget_for(self, department: str) -> Any | None:
        """按部门取预算；查不到返回 None（R014 判 WARN 转人工）。"""


# --------------------------------------------------------------------------
# 收单锚点
# --------------------------------------------------------------------------


@runtime_checkable
class SubmissionAnchor(Protocol):
    """收单锚点：申报数据的**可信来源**（服务器收单时间 / OA 已审批单据）。

    现状（README 已知限制第一条）：提交日期由申请人手填，同一张超期发票
    改个日期就能让 R003 翻面 —— 机器只能证明"单据自洽"，证明不了"申报真实"。
    生产必须让这些字段**来自服务器收单时间和已审批的 OA 申请单，报销人不可写**。
    """

    def submit_date(self) -> date:
        """服务端收单日期 —— R003 的判定基准，申请人不可写。"""

    def reference(self) -> str:
        """上游单据号（OA 流程号等），进审计日志做交叉核对。"""
