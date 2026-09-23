"""审核单落盘 + 审计日志
========================

两件事，分别存在不同文件里：

1. **审核单** ``data/audits/<audit_id>.json`` —— 当前状态，会被覆写。
2. **审计日志** ``data/audits/<audit_id>.log.jsonl`` —— **只追加，永不修改**。
   制度 6.2 要求"每一次审核动作均须记录……留痕记录只追加，不得修改或删除"。

为什么日志不能和审核单合并：审核单是**状态**，日志是**轨迹**。状态会被改写
（比如从 pending_review 变成 approved），轨迹不能 —— 轨迹一旦能改，
它就失去了作为证据的资格。

关于"哪些历史算数"（一个制度层面的判断）
----------------------------------------
:meth:`AuditStore.history_view` **只把"已批准"的审核单算进查重台账**。

理由：同一张发票确实只能报销一次，但**被驳回且从未入账的发票，报销人有权
改正后重新提交**（比如抬头写错了，改对了再报）。如果连被驳回的记录也算查重
命中，那正确的行为反而会被系统堵死。

反过来说，一旦审核通过、凭证入账，这张发票就"用掉了"，再出现就是重复报销。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import AuditResult, AuditState, Decision
from .rules import HistoryHit

AUDIT_FILE_SUFFIX = ".json"
LOG_FILE_SUFFIX = ".log.jsonl"
LOCK_FILE_SUFFIX = ".lock"

#: 哈希链的"创世"前驱哈希。
GENESIS_HASH = "0" * 64

#: 进程内的决策锁注册表：audit_id -> threading.Lock（见 :meth:`AuditStore.decision_lock`）
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.Lock] = {}


def _hash_record(record: dict[str, Any]) -> str:
    """审计事件的哈希：对**不含 hash 字段本身**的记录做规范化 JSON 的 SHA-256。

    ``sort_keys`` 让同一内容的键序差异不影响哈希 —— 哈希要证明的是
    "内容被改过没有"，不是"键顺序被重排过没有"。
    """
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_base_dir() -> Path:
    """默认落盘目录 ``<project_root>/data/audits``。

    刻意在**调用时**而不是 import 时解析 —— import 时读全局配置会让测试
    悄悄写进真实 data/ 目录（见 tests/test_finance.py 的防腐测试）。
    """
    from config.settings import get_settings

    return Path(get_settings().project_root) / "data" / "audits"


class AuditStore:
    """审核单与审计日志的文件存储。

    用法上**永远显式传 base_dir**（测试传 tmp_path），不传才用默认值。
    """

    def __init__(self, base_dir: str | Path | None = None):
        self._base_dir = Path(base_dir) if base_dir else None

    @property
    def base_dir(self) -> Path:
        """**惰性解析**默认目录 —— A1 教训：模块级构造曾把真实 data/ 路径冻结在
        import 时，测试一打 reset 就清空真实审核单。现在首次使用才解析，
        测试改配置即可把整个存储指进临时目录。"""
        return self._base_dir or _default_base_dir()

    # ---- 路径 ----

    def _ensure_dir(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _json_path(self, audit_id: str) -> Path:
        return self.base_dir / f"{_safe_id(audit_id)}{AUDIT_FILE_SUFFIX}"

    def _log_path(self, audit_id: str) -> Path:
        return self.base_dir / f"{_safe_id(audit_id)}{LOG_FILE_SUFFIX}"

    # ---- 审核单 ----

    def save(self, result: AuditResult) -> Path:
        """覆写审核单当前状态。"""
        self._ensure_dir()
        path = self._json_path(result.audit_id)
        path.write_text(
            json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load(self, audit_id: str) -> AuditResult | None:
        """读回审核单。文件不存在或损坏返回 None。"""
        path = self._json_path(audit_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return AuditResult.model_validate(data)
        except Exception:  # noqa: BLE001 —— 损坏的审核单不该让服务崩掉
            return None

    def list_audits(self) -> list[dict[str, Any]]:
        """列出全部审核单的摘要，按创建时间倒序。"""
        if not self.base_dir.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for path in self.base_dir.glob(f"*{AUDIT_FILE_SUFFIX}"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            items.append(
                {
                    "audit_id": data.get("audit_id", path.stem),
                    "state": data.get("state", ""),
                    "suggested_status": data.get("suggested_status", ""),
                    "decision": data.get("decision"),
                    "applicant": (data.get("request") or {}).get("applicant", ""),
                    "department": (data.get("request") or {}).get("department", ""),
                    "expense_type": (data.get("request") or {}).get("expense_type", ""),
                    "amount": (data.get("request") or {}).get("amount"),
                    "invoice_number": (data.get("invoice") or {}).get("invoice_number", ""),
                    "created_at": data.get("created_at", ""),
                }
            )
        items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return items

    # ---- 审计日志（只追加 + 哈希链） ----

    def append_log(self, audit_id: str, event: dict[str, Any]) -> None:
        """追加一条审计事件。**只写不改**，文件用 ``.log.jsonl``（每行一个 JSON）。

        每条事件带 ``seq / prev / hash`` 组成**哈希链**：``hash`` 是本条内容
        （含前条哈希）的 SHA-256。事后改动任何一行，:meth:`verify_log` 都能
        指出从哪一条断的。"只追加"过去只是**约定**（能写就能改），哈希链把它
        变成**可检测**的保证 —— 会计档案场景里，"改了会被发现"比"不许改"更实在。
        """
        self._ensure_dir()
        path = self._log_path(audit_id)
        prev_hash, seq = self._chain_tail(audit_id)
        record = {
            # 统一 UTC：audit 单的 decided_at / created_at 也是 UTC。
            # 审计时间线混两种时区格式，复盘对表时必然困惑。
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seq": seq,
            "prev": prev_hash,
            **event,
        }
        record["hash"] = _hash_record(record)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _chain_tail(self, audit_id: str) -> tuple[str, int]:
        """哈希链的链尾：(前一条的 hash, 下一条的 seq)。演示规模下直接重扫。"""
        events = self.read_log(audit_id)
        if not events:
            return GENESIS_HASH, 1
        last = events[-1]
        return str(last.get("hash") or GENESIS_HASH), int(last.get("seq") or len(events)) + 1

    def verify_log(self, audit_id: str) -> dict[str, Any]:
        """校验审计日志哈希链的完整性。

        :return: ``{"ok": bool, "checked": int, "broken_at": int | None}``
                 —— ``broken_at`` 是第一条对不上的事件 ``seq``，供排查定位。
        """
        prev = GENESIS_HASH
        checked = 0
        for rec in self.read_log(audit_id):
            checked += 1
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("hash") != _hash_record(body) or rec.get("prev") != prev:
                return {"ok": False, "checked": checked, "broken_at": rec.get("seq", checked)}
            prev = rec.get("hash") or GENESIS_HASH
        return {"ok": True, "checked": checked, "broken_at": None}

    def read_log(self, audit_id: str) -> list[dict[str, Any]]:
        """读回审计轨迹。坏行跳过，不抛异常（轨迹残缺好过整个读不出来）。"""
        path = self._log_path(audit_id)
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    # ---- 查重台账 ----

    def history_view(self) -> "StoreHistoryView":
        """返回只读历史视图，供 R004 / R012 使用。"""
        return StoreHistoryView(self)

    def all_records(self) -> list[AuditResult]:
        """**所有**已提交的审核单（含待审、已驳回）。连号检测用。"""
        out: list[AuditResult] = []
        if not self.base_dir.is_dir():
            return out
        for path in self.base_dir.glob(f"*{AUDIT_FILE_SUFFIX}"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                out.append(AuditResult.model_validate(raw))
            except Exception:  # noqa: BLE001
                continue
        return out

    def approved_records(self) -> list[tuple[AuditResult, dict[str, Any]]]:
        """已批准（已入账）的审核单 —— 这才是"发票用掉了"的判据。"""
        out: list[tuple[AuditResult, dict[str, Any]]] = []
        if not self.base_dir.is_dir():
            return out
        for path in self.base_dir.glob(f"*{AUDIT_FILE_SUFFIX}"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if raw.get("decision") != Decision.APPROVED.value:
                continue
            result = AuditResult.model_validate(raw)
            out.append((result, raw))
        return out

    # ---- 演示用 ----

    def clear(self) -> int:
        """清空全部审核单与日志。**仅供演示重置**，真实系统不该有这个入口。"""
        if not self.base_dir.is_dir():
            return 0
        removed = 0
        for path in self.base_dir.iterdir():
            if path.is_file() and (
                path.name.endswith(AUDIT_FILE_SUFFIX)
                or path.name.endswith(LOG_FILE_SUFFIX)
                or path.name.endswith(LOCK_FILE_SUFFIX)
            ):
                path.unlink()
                removed += 1
        return removed

    # ---- 决策锁 ----

    @contextlib.contextmanager
    def decision_lock(self, audit_id: str):
        """同一审核单的「检查已决定 + 落盘」临界区。

        ``decide()`` 原来是"读内存判断再写盘"——典型的 read-modify-write 竞态：
        两个审核员各持一份旧快照，先后点按钮，**两人都能通过"未决定"的检查**，
        后写覆盖先写。而"谁能批、批了几次"正是这个项目的核心命题。
        这里用「线程锁 + 文件锁」双层把该操作串行化：线程锁管同进程并发
        （uvicorn 单进程多执行流），文件锁管多进程（比如误开了两个 server）。
        """
        key = _safe_id(audit_id)
        with _LOCKS_GUARD:
            tlock = _LOCKS.setdefault(key, threading.Lock())
        with tlock:
            self._ensure_dir()
            fh = (self.base_dir / f"{key}{LOCK_FILE_SUFFIX}").open("a+b")
            try:
                _lock_file(fh)
                try:
                    yield
                finally:
                    _unlock_file(fh)
            finally:
                fh.close()


class StoreHistoryView:
    """:class:`finance.rules.HistoryView` 的文件实现。

    **两个查询走不同的时间范围**，这是刻意的：

    - ``find_invoice``（查重 R004）只看**已批准**的记录。一张被驳回、从未入账的
      发票，报销人有权改正后重新提交 —— 连它也算命中，正确的行为反而被堵死。
    - ``find_same_seller_same_date``（连号检测 R012）看**所有已提交**的记录，
      包括还在待审的。连号检测的是"提交模式"，不是"报销事实"；
      同一批拆单票往往是连着几张一起报的，等它们都批完才发现，
      这个规则就失去意义了。

    每次查询都重新扫盘。演示规模下（几十单）完全够用；
    真实系统应换成带索引的数据库查询 —— 接口不变，这正是 Protocol 的意义。
    """

    def __init__(self, store: AuditStore):
        self._store = store

    def _to_hit(self, result: AuditResult, raw: dict) -> HistoryHit:
        issue = result.invoice.issue_date
        if isinstance(issue, str):
            try:
                issue = date.fromisoformat(issue)
            except ValueError:
                issue = None
        return HistoryHit(
            audit_id=result.audit_id,
            invoice_key=result.invoice.key(),
            invoice_number=result.invoice.invoice_number,
            seller_name=result.invoice.seller_name,
            issue_date=issue,
            decided_at=str(raw.get("decided_at") or ""),
        )

    def _approved_hits(self) -> list[HistoryHit]:
        return [self._to_hit(r, raw) for r, raw in self._store.approved_records()]

    def _submitted_hits(self) -> list[HistoryHit]:
        return [
            HistoryHit(
                audit_id=result.audit_id,
                invoice_key=result.invoice.key(),
                invoice_number=result.invoice.invoice_number,
                seller_name=result.invoice.seller_name,
                issue_date=result.invoice.issue_date,
                decided_at="",
            )
            for result in self._store.all_records()
        ]

    def find_invoice(self, invoice_key: str) -> HistoryHit | None:
        for hit in self._approved_hits():
            if hit.invoice_key == invoice_key:
                return hit
        return None

    def find_same_seller_same_date(
        self, seller_name: str, issue_date: date
    ) -> list[HistoryHit]:
        return [
            h
            for h in self._submitted_hits()
            if h.seller_name == seller_name and h.issue_date == issue_date
        ]


class MemoryHistoryView:
    """内存实现，测试与单次演示用（不落盘）。"""

    def __init__(self, hits: Iterator[HistoryHit] | list[HistoryHit] | None = None):
        self._hits: list[HistoryHit] = list(hits or [])

    def add(self, hit: HistoryHit) -> None:
        self._hits.append(hit)

    def find_invoice(self, invoice_key: str) -> HistoryHit | None:
        return next((h for h in self._hits if h.invoice_key == invoice_key), None)

    def find_same_seller_same_date(
        self, seller_name: str, issue_date: date
    ) -> list[HistoryHit]:
        return [
            h
            for h in self._hits
            if h.seller_name == seller_name and h.issue_date == issue_date
        ]


def _lock_file(fh) -> None:
    """对文件首字节加排他锁（Windows 用 msvcrt，POSIX 用 fcntl）。

    锁 1 个字节的区域就够 —— 这里要的是"同一时刻只有一个决策在进行"，
    不是锁内容。等锁上限 5 秒，超时抛出，不无限挂着。
    """
    fh.seek(0)
    if not fh.read(1):
        fh.write(b"\0")
        fh.flush()
    fh.seek(0)
    try:
        import msvcrt
    except ImportError:  # POSIX
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return
    deadline = time.time() + 5.0
    while True:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if time.time() >= deadline:
                raise TimeoutError("等待审核单决策锁超时（另一进程正在决定这张单）")
            time.sleep(0.05)


def _unlock_file(fh) -> None:
    fh.seek(0)
    try:
        import msvcrt
    except ImportError:  # POSIX
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return
    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


def _safe_id(audit_id: str) -> str:
    """防目录穿越：审核单 id 只允许字母数字下划线连字符。"""
    cleaned = "".join(ch for ch in str(audit_id) if ch.isalnum() or ch in "-_")
    if not cleaned:
        raise ValueError(f"非法审核单 id: {audit_id!r}")
    return cleaned
