"""归档：草稿与档案之间的那道门
==============================

这一组测试守的是《会计档案管理办法》和《会计基础工作规范》里最硬的一条：
**没签过字的东西，不能看起来像签过字的东西。**

所以这里的断言比功能测试更"苛刻"：不是"能不能归档"，而是
"什么情况下必须归档失败"。
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

import pytest

from finance.archive import (
    FINAL_SUFFIX,
    AlreadyArchived,
    Archive,
    ArchiveError,
    NotReadyForArchive,
    SeparationOfDuties,
    SignatureError,
    SignatureMismatch,
    build_statement,
    digest_of,
)
from finance.audit import decide, run_audit
from finance.models import (
    AuditResult,
    AuditState,
    Decision,
    Invoice,
    ReimbursementRequest,
    SuggestedStatus,
)
from finance.store import AuditStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PDF = PROJECT_ROOT / "finance" / "samples" / "pdf"


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------


def make_request(**over) -> ReimbursementRequest:
    base = dict(
        applicant="张三",
        department="技术部",
        expense_type="住宿费",
        amount="1650.00",
        reason="上海客户现场支持",
        submit_date=date(2026, 9, 18),
        city="上海",
        nights=3,
    )
    base.update(over)
    return ReimbursementRequest(**base)


@pytest.fixture
def store(tmp_path) -> AuditStore:
    return AuditStore(base_dir=tmp_path / "audits")


@pytest.fixture
def archive(tmp_path) -> Archive:
    return Archive(base_dir=tmp_path / "archive")


def _run(store, pdf_name="S01_hotel_ok.pdf", **req_over) -> AuditResult:
    pdf = SAMPLES_PDF / pdf_name
    if not pdf.is_file():
        pytest.skip(f"样本票 {pdf_name} 缺失，请先运行 scripts/make_samples.py")

    async def _go():
        return await run_audit(pdf, make_request(**req_over), store=store, narrative_llm=None)

    return asyncio.run(_go())


def _approved(store, **kw) -> AuditResult:
    """跑一张通过的单子并做人工批准 —— 归档的前置状态。"""
    result = _run(store, **kw)
    decide(result, Decision.APPROVED, "李会计", store=store)
    return result


def _record_path(archive: Archive, audit_id: str) -> Path:
    """终稿文件路径 —— 用 glob 找，不在测试里拼后缀常量。"""
    hits = list(archive.base_dir.glob(f"{audit_id}*.json"))
    assert hits, f"没有找到 {audit_id} 的归档文件"
    return hits[0]


# --------------------------------------------------------------------------
# 1. 正常归档
# --------------------------------------------------------------------------


def test_finalize_writes_signed_record(store, archive):
    result = _approved(store)
    record = archive.finalize(result, signed_by="李会计", role="财务复核", store=store)

    assert record["schema"] == "nono.archive.v1"
    assert record["audit_id"] == result.audit_id
    assert record["signature"]["signed_by"] == "李会计"
    assert record["signature"]["role"] == "财务复核"
    assert record["signature"]["statement_hash"] == digest_of(build_statement(result))
    assert record["payload"]["rule_version"]  # 结论要能按当时版本复算
    assert record["payload"]["decision"] == "APPROVED"
    assert record["payload"]["voucher"]["lines"]  # 有账才入档

    assert archive.load(result.audit_id)["record_hash"] == record["record_hash"]
    assert archive.finalized_ids() == [result.audit_id]


def test_archive_record_excludes_llm_narrative(store, archive):
    """模型写的审核意见**不进档案** —— 它给人看，不给会计资料用。"""
    result = _approved(store)
    record = archive.finalize(result, signed_by="李会计", role="财务复核")
    assert "narrative" not in record["payload"]
    assert "narrative" not in json.dumps(record["payload"], ensure_ascii=False)


def test_archived_file_is_read_only(store, archive):
    result = _approved(store)
    archive.finalize(result, signed_by="李会计", role="财务复核")
    path = _record_path(archive, result.audit_id)

    with pytest.raises((PermissionError, OSError)):
        path.write_text("{}", encoding="utf-8")


# --------------------------------------------------------------------------
# 2. 什么情况下必须归档失败
# --------------------------------------------------------------------------


def test_pending_review_cannot_be_archived(store, archive):
    result = _run(store)  # 没做决定
    assert result.decision is None
    with pytest.raises(NotReadyForArchive):
        archive.finalize(result, signed_by="李会计", role="财务复核")


def test_rejected_cannot_be_archived(store, archive):
    result = _run(store, "S02_hotel_over_limit.pdf", amount="2400.00")
    decide(result, Decision.REJECTED, "李会计", store=store)
    with pytest.raises(NotReadyForArchive):
        archive.finalize(result, signed_by="李会计", role="财务复核")


def test_approved_without_voucher_cannot_be_archived(archive):
    """批了但没凭证 = 无账可入，不能归档。手工构造一个这种状态。"""
    result = AuditResult(
        invoice=Invoice(invoice_number="123", total="100.00"),
        request=make_request(amount="100.00"),
        state=AuditState.APPROVED,
        decision=Decision.APPROVED,
        operator="李会计",
        suggested_status=SuggestedStatus.APPROVED,
    )
    assert result.voucher is None
    with pytest.raises(NotReadyForArchive):
        archive.finalize(result, signed_by="李会计", role="财务复核")


def test_role_must_be_in_whitelist(store, archive):
    result = _approved(store)
    with pytest.raises(SignatureError):
        archive.finalize(result, signed_by="李会计", role="出纳")


def test_applicant_cannot_sign_own_claim(store, archive):
    result = _approved(store)
    with pytest.raises(SeparationOfDuties):
        archive.finalize(result, signed_by="张三", role="财务复核")


def test_signature_binds_content(store, archive):
    result = _approved(store)
    good = digest_of(build_statement(result))

    with pytest.raises(SignatureMismatch):
        archive.finalize(
            result, signed_by="李会计", role="财务复核", statement_hash="0" * 64
        )
    # 签对了才让过
    record = archive.finalize(
        result, signed_by="李会计", role="财务复核", statement_hash=good
    )
    assert archive.verify_signature(record) == (True, "")


def test_duplicate_archive_is_refused(store, archive):
    result = _approved(store)
    archive.finalize(result, signed_by="李会计", role="财务复核")
    with pytest.raises(AlreadyArchived):
        archive.finalize(result, signed_by="李会计", role="财务复核")


def test_broken_audit_log_blocks_archive(store, archive):
    """轨迹链断了就拒绝归档 —— 证据不完整时不入档。"""
    result = _approved(store)
    log_path = store.base_dir / f"{result.audit_id}.log.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "tampered"}, ensure_ascii=False) + "\n")

    with pytest.raises(ArchiveError) as exc:
        archive.finalize(result, signed_by="李会计", role="财务复核", store=store)
    assert "断裂" in str(exc.value)


# --------------------------------------------------------------------------
# 3. 档案链与篡改检测
# --------------------------------------------------------------------------


def test_chain_links_records_in_order(store, archive):
    first = _approved(store)
    # 第二张：同一张票的第二次提交（会命中查重 R004），走人工推翻批准
    second = _run(store, "S01_hotel_ok.pdf", reason="补报")
    decide(
        second,
        Decision.APPROVED,
        "李会计",
        store=store,
        override_reason="重复申请已电话确认撤销其中一笔，本单为补录",
    )

    rec1 = archive.finalize(first, signed_by="李会计", role="财务复核", store=store)
    rec2 = archive.finalize(second, signed_by="李会计", role="财务复核", store=store)

    assert rec1["seq"] == 1 and rec2["seq"] == 2
    assert rec2["prev_hash"] == rec1["record_hash"]
    assert archive.verify_chain()["ok"] is True
    assert archive.verify_chain()["checked"] == 2


def test_tampered_record_is_detected(store, archive):
    result = _approved(store)
    archive.finalize(result, signed_by="李会计", role="财务复核")
    path = _record_path(archive, result.audit_id)

    # 模拟绕过只读位的人（真实系统里这需要更高权限）
    path.chmod(0o644)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"]["invoice"]["total"] = "0.01"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    report = archive.verify_chain()
    assert report["ok"] is False
    assert report["broken_at"] == result.audit_id
    assert "record_hash" in report["reason"]


def test_tampered_signature_statement_is_detected(store, archive):
    result = _approved(store)
    archive.finalize(result, signed_by="李会计", role="财务复核")
    path = _record_path(archive, result.audit_id)

    path.chmod(0o644)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["signature"]["statement_hash"] = "f" * 64
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    ok, why = archive.verify_signature(data)
    assert ok is False
    assert "签名内容" in why


# --------------------------------------------------------------------------
# 4. 结构性：只有 archive 能写终稿
# --------------------------------------------------------------------------


def test_only_archive_module_writes_final_records():
    """全仓只有 finance/archive.py 知道终稿文件的存在。

    和 test_finance.py 里"只有 audit.py 能改状态"同一套结构性断言：
    把"谁能写档案"收敛到一个模块，出了事才查得到是哪段代码干的。
    """
    offenders: list[str] = []
    skip_parts = {".venv", "data", "logs", "tests", "__pycache__", ".git"}
    for py in PROJECT_ROOT.rglob("*.py"):
        if set(py.parts) & skip_parts:
            continue
        if py.name == "archive.py":
            continue
        if FINAL_SUFFIX in py.read_text(encoding="utf-8"):
            offenders.append(str(py.relative_to(PROJECT_ROOT)))
    assert not offenders, f"这些模块碰了终稿文件：{offenders}"
