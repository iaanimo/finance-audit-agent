"""收件箱：已读 ≠ 已处理
========================

这个文件只测一件事的边界：**消息的两种状态不能混用。**

打开消息是"看"，做出决定才是"办完"。把两者合成一个 ``read`` 字段，
待办数就会永远失真 —— 人批完单，通知还挂着未读，系统看起来永远欠着一堆事。
"""

from __future__ import annotations

import pytest

from agent.inbox import Inbox


@pytest.fixture
def inbox(tmp_path) -> Inbox:
    return Inbox(base_dir=tmp_path / "inbox")


def _send(inbox: Inbox, *, to: str = "审核台", audit_id: str = "a1", kind: str = "draft_ready"):
    return inbox.send(
        to=to,
        kind=kind,
        audit_id=audit_id,
        subject="主题",
        body="正文",
        dedupe_key=f"{audit_id}:{kind}",
    )


def test_new_message_is_unread_and_open(inbox):
    _send(inbox)
    assert inbox.count_unread("审核台") == 1
    assert inbox.count_open("审核台") == 1


def test_reading_does_not_close_the_todo(inbox):
    msg = _send(inbox)
    assert inbox.mark_read("审核台", msg["id"]) is True
    assert inbox.count_unread("审核台") == 0
    assert inbox.count_open("审核台") == 1  # 看过了，但还没办


def test_resolve_closes_multiple_recipients_of_the_same_audit(inbox):
    _send(inbox, to="审核台")
    _send(inbox, to="张三", kind="supplement")
    _send(inbox, to="审核台", audit_id="a2", kind="review_required")

    assert inbox.resolve_for_audit("a1", by="李会计") == 2

    assert inbox.count_open("审核台") == 1  # a2 不受影响
    assert inbox.count_open("张三") == 0
    closed = inbox.messages_for("审核台", open_only=True)
    assert [m["audit_id"] for m in closed] == ["a2"]


def test_resolve_records_who_and_is_idempotent(inbox):
    _send(inbox)
    assert inbox.resolve_for_audit("a1", by="李会计") == 1
    assert inbox.resolve_for_audit("a1", by="李会计") == 0  # 第二次没有可处理的

    msg = inbox.messages_for("审核台")[0]
    assert msg["resolved_by"] == "李会计"
    assert msg["resolved_at"]


def test_unknown_kind_is_rejected(inbox):
    with pytest.raises(ValueError):
        inbox.send(to="审核台", kind="pay_now", audit_id="a1", subject="s", body="b")
