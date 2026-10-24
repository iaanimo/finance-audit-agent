"""编排 —— 从"一张票进来"到"下一步谁该干什么"
================================================

这是"规则引擎"与"agent"之间那条缝上的东西。它做三件事，**不做第四件**：

1. 调 :func:`finance.audit.run_audit`（判定与状态推进交给 finance，一行都不重复实现）；
2. 看结论，决定**程序性动作**：通知审核台 / 向申请人要材料；
3. 每一步都落审计日志，带 actor（谁触发的）、以及模型/提示词版本（若调了模型）。

第四件事故意不做：**它不碰批准、不碰付款、不碰归档。**
这三个动作在 :mod:`agent.gate` 里是 L3，agent 的上限是 L2，越级直接抛异常。
所以"agent 会不会自己把单子批了"这个问题，在代码结构上就不成立 ——
不是靠提示词，是靠它手上没有那把钥匙。

一个诚实的说明：agent 事件排在 audit 阶段之后
---------------------------------------------
``run_audit`` 内部不认识 actor，它生成 audit_id 之后才落盘。所以
``agent.run.started`` 只能等它返回、拿到 audit_id 之后补记 —— 日志里 seq 排在
``validated / duplicate_checked / ...`` 之后。

为什么不改 ``run_audit`` 加一个 actor 参数：那要动判定内核的签名，而这一层的主张
正是"内核不动"。补记的代价是时间线顺序稍微不自然；收益是 finance 的 206 个用例
一个都不用改。日志哈希链保证补记的内容事后也改不了，这个代价可以接受 ——
写在 docstring 里，比藏起来强。

可重放，但**不重跑判定**
-----------------------
:meth:`AgentRuntime.recover` 只补"没发出去的通知"，绝不重新调用 ``run_audit``。
理由很硬：重跑会重新抽票、重新跑 19 条规则、重写 findings ——
那等于把已经发生过的结论再算一遍并覆盖掉，审计上叫篡改，不叫恢复。
恢复的正确含义是：**判定已经落盘且不可变，只有程序性动作允许补做。**
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finance.audit import run_audit
from finance.models import AuditResult, AuditState, ReimbursementRequest
from finance.policy import PolicyBundle
from finance.store import AuditStore

from .actors import Actor, agent_actor
from .gate import Action, Gate, OutOfBoundsAction, spec
from .inbox import Inbox

#: 审核台的"收件人"标识。不是某个人 —— 是一个队列。
#: 真实系统这里应该是角色/组，不是自然人；写死一个名字会造成"发给谁"的伪精确。
REVIEW_DESK = "审核台"

#: 日志事件名。前缀 ``agent.`` 让"这条轨迹里哪些是代理做的"一眼可查。
EV_RUN_STARTED = "agent.run.started"
EV_RUN_FINISHED = "agent.run.finished"
EV_ACTION = "agent.action"


@dataclass
class Step:
    """一次已执行的程序性动作。"""

    action: str
    outcome: str  # sent / deduped / denied
    to: str = ""
    detail: str = ""


@dataclass
class RunReport:
    """一次 run 的结果摘要。给 CLI、测试、以及未来的 UI 时间线用。"""

    audit_id: str
    state: str
    suggested_status: str
    actions: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    recovered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "state": self.state,
            "suggested_status": self.suggested_status,
            "actions": self.actions,
            "steps": [s.__dict__ for s in self.steps],
            "recovered": self.recovered,
        }


class AgentRuntime:
    """单票编排器。**无状态**（状态全在 store 里），所以可以随进程重启。

    刻意不叫 "worker" / "queue"：这一版没有队列、没有后台线程。
    事件源（目录轮询/ERP 队列）下一版再加，它们只负责调用 :meth:`submit`。
    """

    def __init__(
        self,
        *,
        store: AuditStore,
        inbox: Inbox | None = None,
        gate: Gate | None = None,
        policy: PolicyBundle | None = None,
        agent: Actor | None = None,
    ):
        self.store = store
        self.gate = gate or Gate()
        self.policy = policy
        self.agent = agent or agent_actor()
        # 收件箱默认跟着 store 的数据目录走（data/inbox），演示时不用另配
        self.inbox = inbox or Inbox(base_dir=Path(store.base_dir).parent / "inbox")

    # ---- 主流程 ----

    async def submit(
        self,
        invoice_path: str | Path,
        request: ReimbursementRequest,
        *,
        actor: Actor | None = None,
        narrative_llm: Any = None,
        **run_kwargs: Any,
    ) -> RunReport:
        """受理一张票：跑判定 → 分派动作 → 留痕。

        ``narrative_llm`` 默认 ``None``（模板叙述）。**这是刻意的默认值**：
        演示与测试都不该依赖外部模型；要接模型时显式传 ``"auto"``，
        接了就必然在日志里留下 model 字段，便于追责。

        :raises finance.audit.AuditError: 判定流程本身故障（不吞，交给上层）
        """
        actor = actor or self.agent
        src = Path(invoice_path)
        input_sha256 = _sha256_file(src)

        result = await run_audit(
            src,
            request,
            store=self.store,
            policy=self.policy,
            narrative_llm=narrative_llm,
            **run_kwargs,
        )

        # agent 事件的"补记"（为什么晚上一步，见模块 docstring）
        self.store.append_log(
            result.audit_id,
            {
                "event": EV_RUN_STARTED,
                **actor.log_fields(),
                "input": src.name,
                "input_sha256": input_sha256,
                "narrative_source": result.narrative_source.value,
            },
        )

        steps = self._dispatch(result, actor)
        self._finish(result, actor, steps, recovered=False)
        return RunReport(
            audit_id=result.audit_id,
            state=result.state.value,
            suggested_status=result.suggested_status.value,
            actions=[s.action for s in steps],
            steps=steps,
        )

    def recover(self) -> list[RunReport]:
        """断点恢复：给"跑到一半断了"的单子补发通知。

        判定条件很具体 —— 停在 ``pending_review``、没人做过决定、
        日志里没有 ``agent.run.finished``。三条同时满足才说明"人还没接手，
        而 agent 的动作没走完"。

        **不重跑判定**（见模块 docstring）。动作本身幂等（inbox 去重），
        所以跑几遍都不会给申请人刷屏。
        """
        reports: list[RunReport] = []
        for result in self.store.all_records():
            if result.decision is not None:
                continue
            if result.state is not AuditState.PENDING_REVIEW:
                continue
            events = self.store.read_log(result.audit_id)
            if any(e.get("event") == EV_RUN_FINISHED for e in events):
                continue
            steps = self._dispatch(result, self.agent)
            self._finish(result, self.agent, steps, recovered=True)
            reports.append(
                RunReport(
                    audit_id=result.audit_id,
                    state=result.state.value,
                    suggested_status=result.suggested_status.value,
                    actions=[s.action for s in steps],
                    steps=steps,
                    recovered=True,
                )
            )
        return reports

    def timeline(self, audit_id: str) -> list[dict[str, Any]]:
        """把轨迹转成"谁做了什么"的时间线，供 UI 与排查用。

        ``actor_type`` 的判定是**启发式**（agent.* 事件来自 agent；
        decided / prefill_confirmed 来自人；stage / submitted 来自系统），
        但每一项都能在日志里找到原始事件 —— 展示层可以做，判定层不行。
        """
        out: list[dict[str, Any]] = []
        for event in self.store.read_log(audit_id):
            name = str(event.get("event", ""))
            if name.startswith("agent."):
                kind = str(event.get("actor_type") or "agent")
            elif name in ("decided", "prefill_confirmed"):
                kind = "human"
            else:
                kind = "system"
            out.append({**event, "actor_type": kind})
        return out

    # ---- 分派 ----

    def _dispatch(self, result: AuditResult, actor: Actor) -> list[Step]:
        """按结论决定程序性动作。三条分支，互相排斥。"""
        blocking = result.blocking_findings
        warnings = result.warning_findings

        if blocking:
            return [
                self._act(
                    Action.NOTIFY_REVIEWER,
                    actor,
                    result,
                    to=REVIEW_DESK,
                    kind="review_required",
                    dedupe_key=f"{result.audit_id}:review_required",
                    subject=f"【待复核·驳回项】{result.request.applicant} "
                    f"{result.request.expense_type} 单",
                    body=_blocking_body(result),
                )
            ]

        if warnings:
            applicant = result.request.applicant.strip() or "申请人"
            return [
                self._act(
                    Action.REQUEST_INFO,
                    actor,
                    result,
                    to=applicant,
                    kind="supplement",
                    dedupe_key=f"{result.audit_id}:supplement",
                    subject=f"【需补材料】{result.request.expense_type} 报销单",
                    body=_warning_body(result),
                )
            ]

        return [
            self._act(
                Action.NOTIFY_REVIEWER,
                actor,
                result,
                to=REVIEW_DESK,
                kind="draft_ready",
                dedupe_key=f"{result.audit_id}:draft_ready",
                subject=f"【待签批·草稿】{result.request.applicant} "
                f"{result.request.expense_type} 单",
                body=_draft_ready_body(result),
            )
        ]

    def _act(
        self,
        action: Action,
        actor: Actor,
        result: AuditResult,
        *,
        to: str,
        kind: str,
        dedupe_key: str,
        subject: str,
        body: str,
    ) -> Step:
        """执行一个 L1 动作：过闸门 → 投递（幂等）→ 落审计日志。

        越级不抛给调用方，而是**记一条拒绝证据后返回 denied** ——
        一次越权尝试值得记录，但不该让整单处理失败（人还要接手）。
        """
        try:
            self.gate.check(action, actor)
        except OutOfBoundsAction as exc:
            self.store.append_log(result.audit_id, exc.log_fields())
            return Step(action=action.value, outcome="denied", to=to, detail=str(exc))

        already = (
            self.inbox.find(to=to, audit_id=result.audit_id, kind=kind, dedupe_key=dedupe_key)
            is not None
        )
        message = self.inbox.send(
            to=to,
            kind=kind,
            audit_id=result.audit_id,
            subject=subject,
            body=body,
            dedupe_key=dedupe_key,
        )
        outcome = "deduped" if already else "sent"
        self.store.append_log(
            result.audit_id,
            {
                "event": EV_ACTION,
                "action": action.value,
                "tier": int(spec(action).tier),
                "result": outcome,
                "to": to,
                "kind": kind,
                "message_id": message.get("id", ""),
                "dedupe_key": dedupe_key,
                **actor.log_fields(),
            },
        )
        return Step(
            action=action.value,
            outcome=outcome,
            to=to,
            detail=f"kind={kind}",
        )

    def _finish(
        self,
        result: AuditResult,
        actor: Actor,
        steps: list[Step],
        *,
        recovered: bool,
    ) -> None:
        self.store.append_log(
            result.audit_id,
            {
                "event": EV_RUN_FINISHED,
                **actor.log_fields(),
                "state": result.state.value,
                "actions": [s.action for s in steps],
                "outcomes": {s.action: s.outcome for s in steps},
                "recovered": recovered,
            },
        )


# --------------------------------------------------------------------------
# 消息正文 —— 全部模板拼装，**不经过模型**
# --------------------------------------------------------------------------
#
# 为什么不调 LLM 生成催办文案：一旦是模型生成的自由文本，它就落进
# 《人工智能生成合成内容标识办法》的"生成内容"，要么加标识、要么做披露，
# 而这个场景根本不需要模型的文采。模板化还顺带解决了另一个问题：
# 通知里的每一句都能被测试断言，模型生成的不能。


def _blocking_body(result: AuditResult) -> str:
    lines = [
        f"单号：{result.audit_id}　申请人：{result.request.applicant}",
        f"系统建议：{result.suggested_status.value}",
        "命中驳回项（事实性违规）：",
    ]
    for f in result.blocking_findings:
        lines.append(f"  - [{f.rule_id}] 制度{f.clause} {f.title}：{f.message}")
    lines.append("请人工复核后做出决定；系统不会自行批准或驳回。")
    return "\n".join(lines)


def _warning_body(result: AuditResult) -> str:
    lines = [
        f"单号：{result.audit_id}　申请人：{result.request.applicant}",
        "系统预审发现以下疑点，需补充材料后才能继续：",
    ]
    for f in result.warning_findings:
        lines.append(f"  - [{f.rule_id}] 制度{f.clause} {f.title}：{f.message}")
    lines.append("补齐后重新提交本单，系统会重新审核（本单现有结论不会自动改变）。")
    return "\n".join(lines)


def _draft_ready_body(result: AuditResult) -> str:
    voucher = result.voucher
    balanced = "是" if (voucher is not None and voucher.balanced) else "无法生成" if voucher is None else "否"
    lines = [
        f"单号：{result.audit_id}　申请人：{result.request.applicant}",
        "全部规则通过，凭证草稿已生成（借贷平衡：" + balanced + "）。",
        "草稿不是凭证 —— 须经人工审核签名后才能归档入账。",
    ]
    return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    """输入文件的摘要：证明"进来的到底是哪一份文件"。"""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""
