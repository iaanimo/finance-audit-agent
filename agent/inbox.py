"""站内收件箱 —— agent 唯一的"对外动作"落点
============================================

L1 动作（催办、要材料、通知审核台）必须有地方落地，但**不接真通道**才是对的：

1. 真实邮件/企微/短信是集成成本的大头（企业资质、通道审核、模板报备），
   而且每个通道的"送达/撤回/留证"语义都不一样，现在接进来只会让责任边界变模糊；
2. 更重要的 —— 一旦接了真通道，agent 就具备了"对外发消息"的能力，
   这本身就是一个责任面（骚扰、钓鱼、越权通知）。**先用站内收件箱，把能力关在系统里**，
   等通道抽象（`transport`）真正需要时再接，接口不变。

三条硬约束
----------
1. **只追加**：消息写完不改。已读状态另存 ``read_at`` 补记（也是追加一条状态事件？
   不 —— 演示规模下文件级追加一条读取记录足够，见 :meth:`mark_read`）。
2. **幂等**：同一 ``(收件人, 单号, 类型, dedupe_key)`` 只投一次。
   断点恢复会重新走一遍分派逻辑，没有幂等就会给申请人刷一屏催办 ——
   这是"可重放"的前提，不是可选项。
3. **模板化**：消息正文由调用方用模板拼（runtime 负责），
   本模块**不生成任何自由文本**。理由见 agent/runtime.py：模板文案不构成
   《人工智能生成合成内容标识办法》里的"生成内容"，省掉一整套标识与披露义务。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: 收件箱文件后缀。每人一个 JSONL，行内一条消息。
INBOX_SUFFIX = ".inbox.jsonl"

#: 消息类型白名单 —— 只在 agent 能发的那几类里挑，防止以后有人塞进"付款通知"。
KINDS = ("supplement", "review_required", "draft_ready", "reminder")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_name(name: str) -> str:
    """收件人名 -> 文件名。中文保留，路径分隔符之类一律剔除。

    收件人来自申请人/审核台这类外部输入，拼路径前必须净化 ——
    和 store._safe_id 同一类防护。
    """
    cleaned = "".join(ch for ch in str(name) if ch.isalnum() or ch in "-_")
    if not cleaned:
        # 全是符号时用摘要兜底，避免两个人映射到同一个文件
        cleaned = "u" + uuid.uuid5(uuid.NAMESPACE_URL, str(name)).hex[:8]
    return cleaned


class Inbox:
    """文件收件箱。默认落在 ``data/inbox/``，测试一律传 tmp 目录。"""

    def __init__(self, base_dir: str | Path | None = None):
        self._base_dir = Path(base_dir) if base_dir else None

    @property
    def base_dir(self) -> Path:
        """惰性解析 —— 默认目录在**调用时**才取配置，避免 import 时冻结路径。"""
        if self._base_dir is not None:
            return self._base_dir
        from config.settings import get_settings

        return Path(get_settings().data_dir) / "inbox"

    def _path(self, who: str) -> Path:
        return self.base_dir / f"{_safe_name(who)}{INBOX_SUFFIX}"

    # ---- 写 ----

    def send(
        self,
        *,
        to: str,
        kind: str,
        audit_id: str,
        subject: str,
        body: str,
        dedupe_key: str = "",
    ) -> dict[str, Any]:
        """投一条消息。**幂等**：同键已存在则返回既有消息，不重复投。

        :param dedupe_key: 幂等键。runtime 用 ``f"{audit_id}:{kind}"``，
            这样"崩溃后重放"不会给同一个人发第二遍同样的催办。
        :raises ValueError: ``kind`` 不在白名单（宁可报错，也不许塞进资金类通知）
        """
        if kind not in KINDS:
            raise ValueError(f"未知消息类型 {kind!r}，允许：{KINDS}")
        if not str(to).strip():
            raise ValueError("收件人不能为空")
        if not str(audit_id).strip():
            raise ValueError("消息必须挂在一张审核单上 —— 脱离单据的通知无法追溯")

        existing = self.find(to=to, audit_id=audit_id, kind=kind, dedupe_key=dedupe_key)
        if existing is not None:
            return existing

        record = {
            "id": uuid.uuid4().hex[:10],
            "ts": _now(),
            "to": to,
            "kind": kind,
            "audit_id": audit_id,
            "subject": subject,
            "body": body,
            "dedupe_key": dedupe_key,
            "read_at": None,
        }
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self._path(to).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def mark_read(self, who: str, message_id: str) -> bool:
        """把某条消息标为已读。**追加一条状态行**，不原地改文件 ——
        与审计日志同一原则：轨迹不覆盖，只追加。"""
        msg = self._read_all(who)
        target = next((m for m in msg if m.get("id") == message_id), None)
        if target is None or target.get("read_at"):
            return False
        target["read_at"] = _now()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self._path(who).open("a", encoding="utf-8") as fh:
            # 用同 id 追加一条"已读"更新行；read_all 合并时后写覆盖先写
            fh.write(json.dumps(target, ensure_ascii=False) + "\n")
        return True

    # ---- 读 ----

    def _read_all(self, who: str) -> list[dict[str, Any]]:
        path = self._path(who)
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 坏行跳过：收件箱残缺好过整个读不出来
        # 同一 id 的后写覆盖先写（mark_read 靠这个生效）
        merged: dict[str, dict[str, Any]] = {}
        for msg in out:
            merged[msg.get("id", "")] = msg
        return list(merged.values())

    def find(
        self, *, to: str, audit_id: str, kind: str, dedupe_key: str = ""
    ) -> dict[str, Any] | None:
        """按幂等键查已有消息。"""
        for msg in self._read_all(to):
            if (
                msg.get("audit_id") == audit_id
                and msg.get("kind") == kind
                and msg.get("dedupe_key", "") == dedupe_key
            ):
                return msg
        return None

    def messages_for(self, who: str, unread_only: bool = False) -> list[dict[str, Any]]:
        """某人收件箱，按时间倒序。"""
        msgs = self._read_all(who)
        if unread_only:
            msgs = [m for m in msgs if not m.get("read_at")]
        msgs.sort(key=lambda m: str(m.get("ts", "")), reverse=True)
        return msgs

    def count_unread(self, who: str) -> int:
        return len(self.messages_for(who, unread_only=True))
