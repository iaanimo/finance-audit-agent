"""终稿归档 —— 草稿与档案之间的那道门
=====================================

整个项目里，**"批准"和"入账"是两件事**，中间还隔着签名。这个模块专门管后者。

为什么必须单独一个模块
----------------------
改造成 agent 之前，凭证草稿跟审核单一起躺在 ``data/audits/*.json`` 里 ——
草稿和终稿在同一个文件、同一个生命周期。合规上这是有问题的：

- 《会计基础工作规范》：记账凭证须经**审核签名**；
- 《会计档案管理办法》：电子会计档案须可追溯、防篡改、可备份；
- 《电子签名法》：只有"可靠电子签名"才与手写签名同等效力。

草稿没有签名，所以它**不是凭证**。把草稿放在档案目录里，等于让"没签过字的东西"
看起来像签过字的东西 —— 这是审计里最常见、也最要命的混同。

所以本模块只做一件事：**人签了字，才把确定性内容搬进只读档案区；没签，进不来。**

档案正本里装什么、不装什么
--------------------------
装：单据字段、19 条规则的逐条结论 + 证据、凭证分录、系统建议、人工决定、
    规则版本与代码引用、签名信息、归档当时审计轨迹的链头。

**不装 narrative（审核意见那段中文）。** 那是模型给人看的解释，它的价值在"帮助
理解"，它的风险在"被当成事实"。一旦它进档案，它就成了会计资料的一部分 ——
于是《人工智能生成合成内容标识办法》的标识义务、以及"模型编造内容进了档案"
的信任问题，全部落到这个项目头上。不装，两个问题一起消失。

链条
----
每条档案记录带 ``seq / prev_hash / record_hash``，形成**跨单据的档案链**
（store 的哈希链是**单张单子**的轨迹链，两者互补）：
- 单票轨迹链回答"这张单被谁改过"；
- 档案链回答"已归档的这批东西，有没有被抽换或增删"。

:meth:`Archive.verify_chain` 一次校验两层：记录哈希 + 签名声明的内容哈希。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .erp_api import CODE_REF, RULE_VERSION
from .models import AuditResult, Decision
from .store import _lock_file, _unlock_file

#: 档案文件后缀。**全仓只有本模块写它**（tests 有结构性断言）。
FINAL_SUFFIX = ".final.json"

#: 档案记录的 schema 版本。改结构必须升它，否则历史档案没法按版本解释。
SCHEMA = "nono.archive.v1"

#: 档案链的创世前驱（与 store 的 GENESIS_HASH 同构）。
GENESIS_HASH = "0" * 64

#: 允许签字的岗位。生产应接 RBAC/SSO（见 finance/interfaces.py），
#: 这里先做"岗位必须在白名单里"的最小校验 —— 不冒充权限系统。
ALLOWED_SIGNER_ROLES = ("财务复核", "会计主管", "财务负责人")


class ArchiveError(Exception):
    """归档失败。一律不放行 —— 归档宁可不做，也不做半截。"""


class NotReadyForArchive(ArchiveError):
    """单据状态不满足归档条件（没批准 / 没凭证）。"""


class AlreadyArchived(ArchiveError):
    """同一张单只能有一个终稿。重复归档 = 有人在重放流程。"""


class SignatureError(ArchiveError):
    """签名信息不完整或岗位不合规。"""


class SignatureMismatch(SignatureError):
    """签名声明的哈希与档案正本内容不符 —— 签的和存下的不是同一份。"""


class SeparationOfDuties(ArchiveError):
    """申请人给自己的单子签归档 —— 不相容职务分离的直接违背。"""


class Signature(BaseModel):
    """一次归档签名。

    刻意**不实现密码学签名**：本项目没有 UKey/SDK，假装做了才是撒谎。
    这里做的是签名该有的**结构**（谁、什么岗位、用什么方式、签的是哪份内容），
    以及"内容与签名绑定"的校验 —— 换成真签名时，只需要把 ``method`` 换成
    真实机制并补验签函数，接口不变。README 的"合规边界"里要如实写明这一点。
    """

    signed_by: str = Field(description="签名人")
    role: str = Field(description="岗位，必须在 ALLOWED_SIGNER_ROLES 内")
    method: str = Field(default="password+otp", description="认证方式（生产换 UKey/CA）")
    signed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="签名时间（UTC）"
    )
    statement_hash: str = Field(
        default="", description="被签内容的规范化哈希；由 finalize 计算或验算"
    )

    @field_validator("signed_by", "role", "method")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("签名信息不能为空 —— 匿名归档没有意义")
        return str(v).strip()


def _canonical(obj: Any) -> str:
    """规范化 JSON：键排序、不转义中文。哈希要绑定**内容**，不绑定键顺序。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def digest_of(obj: Any) -> str:
    """内容摘要（SHA-256）。档案记录与签名都用它。"""
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def build_statement(result: AuditResult) -> dict:
    """档案**正本内容**。纯函数、确定性、零 LLM —— 同样的单子永远得到同样的哈希。

    注意这里没有 ``narrative``（理由见模块 docstring），也没有 ``state``
    （状态是过程，档案要的是结论与凭据）。
    """
    return {
        "invoice": result.invoice.model_dump(mode="json"),
        "request": result.request.model_dump(mode="json"),
        "findings": [f.model_dump(mode="json") for f in result.findings],
        "voucher": result.voucher.model_dump(mode="json") if result.voucher else None,
        "suggested_status": result.suggested_status.value,
        "decision": result.decision.value if result.decision else None,
        "operator": result.operator,
        "override_reason": result.override_reason,
        "decided_at": result.decided_at.isoformat() if result.decided_at else None,
        "rule_version": RULE_VERSION,
        "code_ref": CODE_REF,
    }


class Archive:
    """只读档案区。写一次，之后改不动（文件置只读 + 哈希链可检测）。"""

    def __init__(self, base_dir: str | Path | None = None):
        self._base_dir = Path(base_dir) if base_dir else None

    @property
    def base_dir(self) -> Path:
        """惰性解析默认目录：``<data>/archive``。测试一律传 tmp。"""
        if self._base_dir is not None:
            return self._base_dir
        from config.settings import get_settings

        return Path(get_settings().data_dir) / "archive"

    def _path(self, audit_id: str) -> Path:
        # 与 store 同一套 id 净化：audit_id 来自外部，拼路径前必须过滤
        cleaned = "".join(ch for ch in str(audit_id) if ch.isalnum() or ch in "-_")
        if not cleaned:
            raise ValueError(f"非法审核单 id: {audit_id!r}")
        return self.base_dir / f"{cleaned}{FINAL_SUFFIX}"

    # ---- 写 ----

    def finalize(
        self,
        result: AuditResult,
        *,
        signed_by: str,
        role: str,
        method: str = "password+otp",
        statement_hash: str = "",
        store: Any = None,
    ) -> dict:
        """把已批准的单子写成终稿档案。

        :param statement_hash: 签名方声明"我签的是这个哈希"。留空则用当前内容计算
            （演示/单机场景）；真实签名流程应先在签名设备上对
            :func:`build_statement` 的结果签名，再把哈希传进来 —— 这样"签的内容"
            与"存档的内容"必须一致，才检测得出来被调包。
        :param store: 传了就顺带核对单票轨迹链，并要求归档时的链头一起入档。
            轨迹断了就拒绝归档 —— **证据不完整时不归档**。

        :raises NotReadyForArchive: 未批准 / 无凭证
        :raises SeparationOfDuties: 签名人正是申请人
        :raises SignatureError: 岗位不合法
        :raises SignatureMismatch: 声明哈希与内容不符
        :raises AlreadyArchived: 重复归档
        :raises ArchiveError: 轨迹链断裂
        """
        if result.decision is not Decision.APPROVED:
            raise NotReadyForArchive(
                f"审核单 {result.audit_id} 尚未被人工批准，不能入档 —— "
                "草稿与驳回单都不是会计档案"
            )
        if result.voucher is None:
            raise NotReadyForArchive(
                f"审核单 {result.audit_id} 没有凭证，不能入档（无账可入）"
            )
        if not str(signed_by).strip():
            raise SignatureError("归档必须有签名人")
        if role not in ALLOWED_SIGNER_ROLES:
            raise SignatureError(
                f"岗位「{role}」无归档签字权，允许：{'/'.join(ALLOWED_SIGNER_ROLES)}"
            )
        if str(signed_by).strip() == str(result.request.applicant).strip():
            raise SeparationOfDuties(
                f"「{signed_by}」既是申请人又是签批人 —— "
                "不相容职务分离不允许同一人完成申请与签批"
            )

        payload = build_statement(result)
        content_digest = digest_of(payload)
        if statement_hash and statement_hash != content_digest:
            raise SignatureMismatch(
                "签名声明的内容哈希与档案正本不符 —— 签名后被改过，或签的不是这份"
            )

        # 轨道证据：没传 store 就跳过（测试/离线场景），传了就必须是完好的
        log_events = 0
        log_head = ""
        if store is not None:
            events = store.read_log(result.audit_id)
            log_events = len(events)
            log_head = str(events[-1].get("hash", "")) if events else ""
            check = store.verify_log(result.audit_id)
            if not check.get("ok"):
                raise ArchiveError(
                    f"审核单 {result.audit_id} 的审计轨迹哈希链在 "
                    f"{check.get('broken_at')} 处断裂 —— 证据不完整，拒绝归档"
                )

        self.base_dir.mkdir(parents=True, exist_ok=True)
        with _archive_lock(self.base_dir):
            path = self._path(result.audit_id)
            if path.exists():
                raise AlreadyArchived(f"审核单 {result.audit_id} 已经归档，不重复入档")

            prev_hash, seq = self._chain_tail()
            # 走一次 Signature 模型：字段校验、时间戳、序列化只有一处真相
            signature = Signature(
                signed_by=signed_by,
                role=role,
                method=method,
                statement_hash=content_digest,
            )
            record = {
                "schema": SCHEMA,
                "audit_id": result.audit_id,
                "seq": seq,
                "prev_hash": prev_hash,
                "archived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "payload": payload,
                "signature": signature.model_dump(mode="json"),
                "audit_log": {"events": log_events, "head_hash": log_head},
            }
            record["record_hash"] = digest_of(record)

            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            try:
                # 置只读：这是"不可变"的第一道（粗）保障，第二道是哈希链。
                # 真实系统应落 WORM 存储或对象锁，并把文件指纹交可信时间戳（TSA）。
                os.chmod(path, stat.S_IREAD)
            except OSError:  # pragma: no cover —— 某些文件系统不支持
                pass
        return record

    # ---- 读 ----

    def load(self, audit_id: str) -> dict | None:
        path = self._path(audit_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list_records(self) -> list[dict]:
        """按 ``seq`` 升序返回全部档案记录。坏文件跳过（残缺好过读不出来）。"""
        if not self.base_dir.is_dir():
            return []
        out: list[dict] = []
        for path in self.base_dir.glob(f"*{FINAL_SUFFIX}"):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        out.sort(key=lambda r: int(r.get("seq", 0)))
        return out

    def finalized_ids(self) -> list[str]:
        return [str(r.get("audit_id", "")) for r in self.list_records()]

    # ---- 校验 ----

    def verify_chain(self) -> dict:
        """校验档案链 + 每条记录的签名绑定。

        :return: ``{"ok": bool, "checked": int, "broken_at": str | None,
                  "signature_ok": bool}``
        """
        prev = GENESIS_HASH
        checked = 0
        for record in self.list_records():
            checked += 1
            body = {k: v for k, v in record.items() if k != "record_hash"}
            if record.get("record_hash") != digest_of(body):
                return _broken(checked, record, "record_hash 不符（内容被改过）")
            if record.get("prev_hash") != prev:
                return _broken(checked, record, "prev_hash 断链（记录被删/被换）")
            ok, why = self.verify_signature(record)
            if not ok:
                return _broken(checked, record, why)
            prev = str(record.get("record_hash") or GENESIS_HASH)
        return {"ok": True, "checked": checked, "broken_at": None, "signature_ok": True}

    def verify_signature(self, record: dict) -> tuple[bool, str]:
        """签名绑定校验：签名声明的哈希 == 档案正本内容哈希。"""
        payload = record.get("payload") or {}
        sig = record.get("signature") or {}
        if not sig.get("signed_by"):
            return False, "档案没有签名人"
        if digest_of(payload) != sig.get("statement_hash"):
            return False, "签名内容与档案正本不符（payload 在签名后被改过）"
        applicant = str((payload.get("request") or {}).get("applicant") or "").strip()
        if applicant and applicant == str(sig.get("signed_by") or "").strip():
            return False, "签名人等于申请人（不相容职务分离）"
        return True, ""

    # ---- 内部 ----

    def _chain_tail(self) -> tuple[str, int]:
        records = self.list_records()
        if not records:
            return GENESIS_HASH, 1
        last = records[-1]
        return str(last.get("record_hash") or GENESIS_HASH), int(last.get("seq", 0)) + 1


@contextlib.contextmanager
def _archive_lock(base: Path):
    """档案目录级排他锁 —— 保住 ``prev_hash`` 的读-写不被打断。

    复用 store 的文件锁实现（同一套 Windows/POSIX 兼容逻辑），
    不另写一份，避免两处实现漂移。
    """
    fh = (base / ".archive.lock").open("a+b")
    try:
        _lock_file(fh)
        try:
            yield
        finally:
            _unlock_file(fh)
    finally:
        fh.close()


def _broken(checked: int, record: dict, why: str) -> dict:
    return {
        "ok": False,
        "checked": checked,
        "broken_at": str(record.get("audit_id") or record.get("seq")),
        "reason": why,
        "signature_ok": False,
    }
