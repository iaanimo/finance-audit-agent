"""动作分级 —— 谁能做哪一步
============================

这个模块是整条"受控"主张的**机械落点**。它回答一个问题：

    这个动作，agent 有没有资格自己发起？

答案不是靠提示词里写"你不许付款"，而是靠一张**动作表**：每个动作有级别，
级别超过 agent 上限的，在**调用点**就抛异常。笼子不是规则写的，是 API surface 决定的 ——
这也是本项目"把 LLM 关进笼子"最硬的那一层：LLM 连付款工具都看不到，谈不上"被拦住"。

分级依据：按"法律责任落在谁头上"，不按金额
-------------------------------------------
上一版清单按金额分（小额自动、大额人工），那是内控视角。法律视角要问的是
"这一步做完，谁的名字会出现在凭证/付款凭证上"，四档：

L0 只读（读票、算规则、查重、查预算）
    不产生任何对外效果。《会计法》不关心谁算了这笔账，只关心结论真不真。
    所以判定内核（finance/rules.py）本来就不该有 agent 的参与 —— 它是 L0 的机器，
    不是 L0 的"代理"。

L1 程序性通知（催办、要材料、通知审核台）
    只发消息，不产生会计后果。风险是"骚扰/被利用钓鱼"，用模板 + 限次 + 受益人限定控制。
    法律上无签字要求，agent 可做。

L2 拟制文书（凭证草稿、补正建议）
    产生"给人看/给人签"的文本，但**本身不是会计凭证**。《会计基础工作规范》要求
    记账凭证经审核签名 —— 草稿没有签名，所以它只是备稿，不是凭证。
    agent 可做，但**必须标 draft，且只能落在草稿表**（见 finance/archive.py 的拆分）。

L3 产生法律后果（批准、付款、归档签字）
    一旦做了，法律责任就落在这个动作上。只有人能做，且要有权限、有留痕、
    不相容职务分离。企业内部控制基本规范、会计法、电子签名法都指向同一件事：
    **签字这件事不能外包给程序**。

新增动作的规矩
--------------
加一个 Action 就必须在 :data:`ACTION_SPECS` 里写明级别和法律依据，
否则 :func:`spec` 会抛 KeyError —— 宁可启动失败，也不许"默认放行"。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .actors import Actor


class Tier(enum.IntEnum):
    """动作级别。数值只用于比较，顺序有意义，不要重排。"""

    L0_READ = 0
    L1_NOTIFY = 1
    L2_DRAFT = 2
    L3_SIGN = 3


class Action(str, enum.Enum):
    """agent 层能发起或需要判断的全部动作。

    刻意**不包含**抽取之外的模型能力 —— 抽取是 finance.extractor 的事，
    agent 只"调用"，不"重新实现"。所有动作用字符串值落日志，便于跨版本回读。
    """

    EXTRACT = "extract"
    EVALUATE = "evaluate"
    DRAFT_VOUCHER = "draft_voucher"
    REQUEST_INFO = "request_info"
    NOTIFY_REVIEWER = "notify_reviewer"
    REMIND = "remind"
    APPROVE = "approve"
    PAY = "pay"
    FILE_ARCHIVE = "file_archive"


@dataclass(frozen=True)
class ActionSpec:
    """一个动作的级别与法律依据。``legal_basis`` 会**原样写进拒绝日志** ——
    让审计员看到的不是"系统拒绝了"，而是"系统依据哪一条拒绝了"。"""

    tier: Tier
    legal_basis: str
    note: str = ""


ACTION_SPECS: dict[Action, ActionSpec] = {
    Action.EXTRACT: ActionSpec(
        Tier.L0_READ,
        "《会计法》只对会计资料真实性负责，不对抽取工具作要求",
        "抽取允许出错，错误由下游 19 条规则抓住；这是分层的目的",
    ),
    Action.EVALUATE: ActionSpec(
        Tier.L0_READ,
        "判定结论必须可复现（零 LLM），agent 只调用不参与",
        "finance.rules 是唯一判定权所在",
    ),
    Action.DRAFT_VOUCHER: ActionSpec(
        Tier.L2_DRAFT,
        "《会计基础工作规范》记账凭证须经审核签名 —— 草稿不是凭证",
        "草稿必须标 draft 且只能落草稿表，人签后才进档案",
    ),
    Action.REQUEST_INFO: ActionSpec(
        Tier.L1_NOTIFY,
        "程序性沟通，不产生会计后果",
        "模板化 + 限次 + 限定受益人，防止变成骚扰/钓鱼工具",
    ),
    Action.NOTIFY_REVIEWER: ActionSpec(
        Tier.L1_NOTIFY,
        "通知审核台，不代替任何人的判断",
        "通知里只放系统结论与规则号，不放模型自由文字",
    ),
    Action.REMIND: ActionSpec(
        Tier.L1_NOTIFY,
        "催办，程序性沟通",
        "同一单同一对象限次，超限转人工",
    ),
    Action.APPROVE: ActionSpec(
        Tier.L3_SIGN,
        "《会计法》单位负责人/会计人员对会计资料真实性负责，签字不可代理",
        "agent 不得代签；只能把单子推到待签",
    ),
    Action.PAY: ActionSpec(
        Tier.L3_SIGN,
        "《企业内部控制基本规范》不相容职务分离：制单、审核、付款不得同人",
        "付款指令是资金动作，必须双人复核；agent 工具表里根本没有这个动作",
    ),
    Action.FILE_ARCHIVE: ActionSpec(
        Tier.L3_SIGN,
        "《会计档案管理办法》归档件须可追溯、防篡改；《电子签名法》可靠电子签名",
        "归档动作由人触发并签名，agent 只能生成档案正本内容（payload）",
    ),
}

#: agent 的能力上限。想放开某一级，**改这一行并在 PR 里写清法律依据**，不要在调用点打补丁。
AGENT_MAX_TIER = Tier.L2_DRAFT


class OutOfBoundsAction(Exception):
    """agent 试图做超出自身级别的动作。

    这不是"使用了错误的方式"，是"这件事根本不该由它做"。
    调用方（runtime）**必须**把这个异常写进审计日志后再继续 ——
    一次越权尝试本身就是证据，不能悄悄吞掉。
    """

    def __init__(self, action: Action, actor: Actor, spec: ActionSpec):
        self.action = action
        self.actor = actor
        self.spec = spec
        super().__init__(
            f"{actor.type.value}「{actor.name}」无权执行 {action.value}"
            f"（L{spec.tier.value}，上限 L{AGENT_MAX_TIER.value}）：{spec.legal_basis}"
        )

    def log_fields(self) -> dict:
        """写进审计日志的字段 —— 拒绝也要留证据。"""
        return {
            "event": "agent.gate.denied",
            "action": self.action.value,
            "tier": int(self.spec.tier),
            "legal_basis": self.spec.legal_basis,
            **self.actor.log_fields(),
        }


def spec(action: Action | str) -> ActionSpec:
    """取动作规格。未知动作直接抛 KeyError —— 不允许"默认放行"。"""
    return ACTION_SPECS[Action(action)]


def is_agent_allowed(action: Action | str) -> bool:
    """agent 能不能发起这个动作（按级别判断）。"""
    return spec(action).tier <= AGENT_MAX_TIER


#: 未来 ``agent/tools.py`` 的注册源：**只从这里取**，L3 动作永远不在里面。
#: 写成元组而不是函数：让它成为常量，测试可以逐项断言。
AGENT_TOOL_ACTIONS: tuple[Action, ...] = tuple(a for a in Action if is_agent_allowed(a))


class Gate:
    """权限闸门。无状态，唯一职责是"越级就抛"。"""

    def check(self, action: Action | str, actor: Actor) -> None:
        """:raises OutOfBoundsAction: agent/system 发起 L3 动作时。"""
        action = Action(action)
        if actor.is_agent and not is_agent_allowed(action):
            raise OutOfBoundsAction(action, actor, spec(action))

    def explain(self, action: Action | str) -> dict:
        """给 UI / 演示用的可读说明（为什么这一步是 agent 做/不是）。"""
        s = spec(action)
        return {
            "action": Action(action).value,
            "tier": int(s.tier),
            "legal_basis": s.legal_basis,
            "agent_allowed": s.tier <= AGENT_MAX_TIER,
            "note": s.note,
        }
