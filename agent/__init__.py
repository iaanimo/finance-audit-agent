"""受控 Agent 层
===============

这一层存在的理由，一句话：**把"人跑批处理"变成"系统自己推进"，但不改变
任何一条判定，也不接管任何一个签字。**

四块积木

- :mod:`agent.actors`  —— 谁是触发者（日志里必须有名字）
- :mod:`agent.gate`    —— 动作分级（谁能做哪一步，越级就抛）
- :mod:`agent.inbox`   —— 站内收件箱（L1 通知的落点，不接真通道）
- :mod:`agent.runtime` —— 编排（调 finance.audit，然后"该通知的通知、该要料的要料"）

与 finance 的边界
-----------------
finance 管"这张单合不合规"，agent 管"下一步谁该干什么"。
agent **只调用** finance 的公开函数，不复制、不绕过、不修改它的状态机：
``run_audit`` 仍然只能把人推到 ``pending_review``，``decide`` 仍然只有人能调。

这个边界由 tests/test_agent_runtime.py 的结构性断言守住：
agent 层不许出现 ``decide(`` / ``store.save(`` / ``.state = AuditState``。
"""

from .actors import AGENT_NAME, Actor, ActorType, agent_actor, human, system
from .gate import (
    AGENT_MAX_TIER,
    AGENT_TOOL_ACTIONS,
    ACTION_SPECS,
    Action,
    ActionSpec,
    Gate,
    OutOfBoundsAction,
    Tier,
    is_agent_allowed,
    spec,
)
from .inbox import Inbox
from .runtime import AgentRuntime, RunReport, Step

__all__ = [
    "AGENT_NAME",
    "Actor",
    "ActorType",
    "agent_actor",
    "human",
    "system",
    "AGENT_MAX_TIER",
    "AGENT_TOOL_ACTIONS",
    "ACTION_SPECS",
    "Action",
    "ActionSpec",
    "Gate",
    "OutOfBoundsAction",
    "Tier",
    "is_agent_allowed",
    "spec",
    "Inbox",
    "AgentRuntime",
    "RunReport",
    "Step",
]
