"""事件源：收件口 → 受理 → done / failed
========================================

一条不变的判据：**事件源比下游更早认出"同一件事"**。
否则重复投放会生成第二张审核单，而第二张会命中自己的查重规则 ——
假阳性比漏判更难解释。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from agent import AgentRuntime, Inbox
from agent.events import DirectorySource, ScanInProgress, ScanLock
from finance.store import AuditStore
from server import _build_request  # 复用唯一一份申请单校验

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PDF = PROJECT_ROOT / "finance" / "samples" / "pdf"

CLAIM = {
    "applicant": "张三",
    "department": "技术部",
    "expense_type": "住宿费",
    "amount": "1650.00",
    "reason": "上海客户现场支持",
    "submit_date": "2026-09-18",
    "city": "上海",
    "nights": "3",
}


@pytest.fixture
def inbound(tmp_path) -> Path:
    root = tmp_path / "inbound"
    root.mkdir()
    return root


@pytest.fixture
def source(inbound) -> DirectorySource:
    return DirectorySource(inbound, build_request=_build_request)


@pytest.fixture
def runtime(tmp_path) -> AgentRuntime:
    return AgentRuntime(
        store=AuditStore(base_dir=tmp_path / "audits"),
        inbox=Inbox(base_dir=tmp_path / "inbox"),
    )


def _drop(root: Path, name: str = "S01_hotel_ok.pdf", claim: dict | None = None) -> Path:
    document = root / name
    shutil.copy(SAMPLES_PDF / name, document)
    payload = CLAIM if claim is None else claim
    (root / f"{Path(name).stem}.claim.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return document


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pair_is_handled_and_quarantined(source, runtime, inbound):
    _drop(inbound)
    reports = asyncio.run(source.poll_once(runtime))

    assert len(reports) == 1
    assert reports[0].state == "pending_review"

    done = list((inbound / "done").iterdir())
    assert len(done) == 1
    assert (done[0] / "S01_hotel_ok.pdf").is_file()
    assert (done[0] / "S01_hotel_ok.claim.json").is_file()

    ledger = source.ledger()
    assert len(ledger) == 1
    assert ledger[0]["audit_id"] == reports[0].audit_id
    # 事件 id 是内容哈希，不是文件名 —— 改名重投也认得出
    assert len(ledger[0]["event_id"]) == 16

    assert runtime.inbox.messages_for("审核台")[0]["audit_id"] == reports[0].audit_id


def test_original_document_is_not_rewritten(source, runtime, inbound):
    document = _drop(inbound)
    before = _sha(document)
    asyncio.run(source.poll_once(runtime))

    kept = next((inbound / "done").iterdir()) / document.name
    assert _sha(kept) == before  # 只搬不改：原件是证据链的根


def test_redelivery_creates_no_second_audit(source, runtime, inbound):
    _drop(inbound)
    first = asyncio.run(source.poll_once(runtime))
    assert len(first) == 1

    # 同一份材料再放一次（模拟重启后残留、或人工重复投放）
    _drop(inbound)
    second = asyncio.run(source.poll_once(runtime))

    assert second == []
    assert len(runtime.store.list_audits()) == 1
    assert len(source.ledger()) == 1  # 账本只有一条：同一件事只记一次
    # 收件口被清空（账本与扫描锁这两个隐藏文件除外），不会堆在那里反复被扫
    leftover = [
        p.name for p in inbound.iterdir() if p.is_file() and not p.name.startswith(".")
    ]
    assert leftover == []


def test_missing_claim_goes_to_failed(source, runtime, inbound):
    shutil.copy(SAMPLES_PDF / "S01_hotel_ok.pdf", inbound / "S01_hotel_ok.pdf")

    assert asyncio.run(source.poll_once(runtime)) == []
    failed = list((inbound / "failed").iterdir())
    assert len(failed) == 1
    assert "缺少申请单" in (failed[0] / "error.txt").read_text(encoding="utf-8")
    # 没有申请单就没有申报人 —— 系统不替人编一个
    assert runtime.store.list_audits() == []


def test_invalid_claim_goes_to_failed(source, runtime, inbound):
    _drop(inbound, claim={**CLAIM, "applicant": "  "})

    assert asyncio.run(source.poll_once(runtime)) == []
    failed = list((inbound / "failed").iterdir())
    error = (failed[0] / "error.txt").read_text(encoding="utf-8")
    assert "申请单不可用" in error
    assert "申请人" in error


def test_broken_document_goes_to_failed(source, runtime, inbound):
    document = inbound / "broken.pdf"
    document.write_bytes(b"not a pdf at all")
    (inbound / "broken.claim.json").write_text(
        json.dumps(CLAIM, ensure_ascii=False), encoding="utf-8"
    )

    assert asyncio.run(source.poll_once(runtime)) == []
    failed = list((inbound / "failed").iterdir())
    assert "受理失败" in (failed[0] / "error.txt").read_text(encoding="utf-8")


def test_non_document_files_are_ignored(source, runtime, inbound):
    (inbound / "notes.txt").write_text("hi", encoding="utf-8")
    (inbound / "readme.claim.json").write_text("{}", encoding="utf-8")

    assert asyncio.run(source.poll_once(runtime)) == []
    assert not (inbound / "failed").exists()
    assert (inbound / "notes.txt").is_file()


def test_builder_is_required():
    with pytest.raises(TypeError):
        DirectorySource("somewhere")  # type: ignore[call-arg]


def test_concurrent_scan_is_refused_not_duplicated(source, runtime, inbound):
    """两轮扫描不能同时跑：幂等键要等处理完才写得进账本，锁必须事前互斥。"""
    _drop(inbound)
    lock = ScanLock(inbound)
    lock.acquire()
    try:
        with pytest.raises(ScanInProgress):
            asyncio.run(source.poll_once(runtime))
    finally:
        lock.release()

    # 被挡住的那次没有受理任何东西；锁释放后正常处理一次，且只有一张单
    assert runtime.store.list_audits() == []
    assert len(asyncio.run(source.poll_once(runtime))) == 1
    assert len(runtime.store.list_audits()) == 1
