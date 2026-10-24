"""Agent 的身份 —— 审计日志里的"谁"
====================================

这件事看着小，其实是整个 agent 层的地基：

**审计日志如果没有 actor，它就只是"发生了什么"，不是"谁让它发生的"。**
而《会计档案管理办法》和《企业内部控制基本规范》要的恰恰是后者 ——
出了事能追到人和系统，"系统自己干的"不是免责理由，是必须能被指认的对象。

所以本模块只做一件事：把"触发者"变成一个可写进日志、可被测试断言的字段。

三类 actor
----------
- ``human``  —— 真正有签字权的人。法律后果落在他们头上。
- ``agent``  —— 本项目的受控代理。**它在日志里必须存在，但在责任上必须不独立存在**：
  它做的每一步要么是只读判定，要么是程序性通知，要么是给人准备的草稿。
- ``system`` —— 非人非模型的触发者（定时扫描、事件源轮询、重试调度）。

为什么不把 agent 当成 system
---------------------------
因为两者在追责上不同。``system`` 出错是"程序 bug"，``agent`` 出错要说清楚
"哪一版模型、哪一版提示词、哪一版规则" —— 这正是 :class:`Actor` 带
``model`` / ``prompt_hash`` 的原因。填了才有追责价值，不填等于放弃解释权。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

#: 本 agent 在日志里的固定名字。名字进日志，改名等于换责任人，不要随便改。
AGENT_NAME = "nono-agent"


class ActorType(str, Enum):
    """触发者类型。字符串值直接进 JSON 日志，**不要改已有取值**（会破坏历史日志的解释）。"""

    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass(frozen=True)
class Actor:
    """一次动作的触发者。

    ``model`` / ``prompt_hash`` 只对 ``agent`` 有意义：
    - ``model``      实际调用的模型标识（如 ``deepseek-chat``）；
    - ``prompt_hash`` 提示词模板的摘要 —— 提示词是"代码"，改了就该留版本号。

    两者都可以为空（纯规则路径根本不调模型），但**调了模型就必须填**，
    由 :func:`agent_actor` 的调用方负责注入。
    """

    name: str
    type: ActorType = ActorType.HUMAN
    model: str = ""
    prompt_hash: str = ""
    #: 人类操作人的岗位（复核/出纳/…），agent 与 system 留空。
    role: str = field(default="")

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("actor 必须有名字 —— 没名字的动作等于没责任人")

    @property
    def is_agent(self) -> bool:
        """是否受权限闸门约束。``system`` 与 ``agent`` 同等受限（都不许替人签字）。"""
        return self.type is not ActorType.HUMAN

    def log_fields(self) -> dict:
        """写进审计日志的字段。空字段不写 —— 日志里出现 ``model: ""`` 比没有更误导。"""
        out: dict = {"actor": self.name, "actor_type": self.type.value}
        if self.role:
            out["actor_role"] = self.role
        if self.model:
            out["model"] = self.model
        if self.prompt_hash:
            out["prompt_hash"] = self.prompt_hash
        return out


def agent_actor(model: str = "", prompt_hash: str = "") -> Actor:
    """本项目的受控代理。日志里固定叫 ``nono-agent``。"""
    return Actor(
        name=AGENT_NAME,
        type=ActorType.AGENT,
        model=model,
        prompt_hash=prompt_hash,
    )


def human(name: str, role: str = "") -> Actor:
    """人类操作人。名字必填 —— 匿名签字在合规上没有意义。"""
    return Actor(name=name, type=ActorType.HUMAN, role=role)


def system(name: str = "scheduler") -> Actor:
    """定时任务 / 事件源等非人非模型触发者。"""
    return Actor(name=name, type=ActorType.SYSTEM)
