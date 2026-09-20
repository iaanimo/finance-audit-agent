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

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

from .models import AuditResult, AuditState, Decision
from .rules import HistoryHit

AUDIT_FILE_SUFFIX = ".json"
LOG_FILE_SUFFIX = ".log.jsonl"


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
        self.base_dir = Path(base_dir) if base_dir else _default_base_dir()

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

    # ---- 审计日志（只追加） ----

    def append_log(self, audit_id: str, event: dict[str, Any]) -> None:
        """追加一条审计事件。**只写不改**，文件用 ``.log.jsonl``（每行一个 JSON）。"""
        self._ensure_dir()
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            **event,
        }
        with self._log_path(audit_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

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
                path.name.endswith(AUDIT_FILE_SUFFIX) or path.name.endswith(LOG_FILE_SUFFIX)
            ):
                path.unlink()
                removed += 1
        return removed


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
        self._approved: list[HistoryHit] | None = None
        self._submitted: list[HistoryHit] | None = None

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
        if self._approved is None:
            self._approved = [
                self._to_hit(r, raw) for r, raw in self._store.approved_records()
            ]
        return self._approved

    def _submitted_hits(self) -> list[HistoryHit]:
        if self._submitted is None:
            hits: list[HistoryHit] = []
            for result in self._store.all_records():
                hits.append(
                    HistoryHit(
                        audit_id=result.audit_id,
                        invoice_key=result.invoice.key(),
                        invoice_number=result.invoice.invoice_number,
                        seller_name=result.invoice.seller_name,
                        issue_date=result.invoice.issue_date,
                        decided_at="",
                    )
                )
            self._submitted = hits
        return self._submitted

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


def _safe_id(audit_id: str) -> str:
    """防目录穿越：审核单 id 只允许字母数字下划线连字符。"""
    cleaned = "".join(ch for ch in str(audit_id) if ch.isalnum() or ch in "-_")
    if not cleaned:
        raise ValueError(f"非法审核单 id: {audit_id!r}")
    return cleaned
