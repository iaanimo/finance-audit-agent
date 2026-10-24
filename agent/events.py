"""事件源 —— 把"丢一张票进目录"变成一次受理
============================================

为什么先做目录轮询，而不是 webhook / 消息队列
---------------------------------------------
这是有意的**从简**，不是没做完：

1. 目录轮询是唯一一种**不依赖第三方可用性**的事件源 —— 演示、测试、CI 全都能跑；
2. 它把"事件"还原成最原始的两件事：**一份发票原件 + 一份申请单**；
3. 换 webhook 时，改的只是本模块的 ``_pairs()``（从目录取，改成从 HTTP 请求体取），
   :meth:`AgentRuntime.submit` 那层完全不动。

目录协议（一个作业 = 两份文件）
-------------------------------

::

    inbound/
      S01_hotel_ok.pdf                 发票原件（.pdf/.xml/.html/.ofd/图片）
      S01_hotel_ok.claim.json          申请单：{"applicant": ..., "amount": ...}

    处理后（原样移走，**不改写内容**）：
    inbound/done/<event_id>/           受理成功
    inbound/failed/<event_id>/         受理失败，附 error.txt —— 等人工处理，不自动重试
    inbound/.processed.jsonl           账本：{event_id, document, audit_id, at}

为什么是两份文件、不是一个"作业信封"
------------------------------------
发票原件是证据链的根（``raw_text`` / ``source_file`` 都指向它）。把它塞进一个
作业 JSON 里会同时破坏两件事：原件不再是原件，作业文件谁都能改。
申请单本来就是独立材料 —— 「原始凭证与报销单分别保管」是会计基础工作规范的要求，
目录结构照这个来。

幂等键 = 文档字节 + 申请单字节的哈希
------------------------------------
同一份材料重复丢进目录，只受理一次。这不是为了省事：**重复受理会产生第二张
审核单，而第二张会命中自己的查重规则 R004，制造一个假阳性**。
事件源必须比下游更早地把"同一件事"认出来。

构造申请单的函数由调用方注入
----------------------------
``build_request`` 必传。理由：申请单的必填校验只有一处（``server.py::_build_request``），
在这里再写一份，两份迟早漂移 —— 而"绕过页面直接 POST"正是服务端那道校验存在的理由，
它不该被一个"文件源自己的宽松版本"绕过去。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from finance.models import ReimbursementRequest

from .runtime import AgentRuntime, RunReport

#: 申请单文件名后缀：``<stem>.claim.json``
CLAIM_SUFFIX = ".claim.json"

#: 收件口认得的文档类型。与 ``server.py::_ALLOWED_SUFFIXES`` 对齐（那边还含 gif/webp/bmp）。
DOCUMENT_SUFFIXES = (".pdf", ".xml", ".html", ".ofd", ".png", ".jpg", ".jpeg")

DONE_DIR = "done"
FAILED_DIR = "failed"
LEDGER_NAME = ".processed.jsonl"

BuildRequest = Callable[[dict], ReimbursementRequest]


@dataclass(frozen=True)
class InboundEvent:
    """一次待受理的事件。event_id 就是幂等键。"""

    event_id: str
    document: Path
    claim: Path
    request: ReimbursementRequest


class DirectorySource:
    """目录事件源。``root`` 下每一对「文档 + 申请单」就是一个待受理作业。"""

    def __init__(
        self,
        root: str | Path,
        *,
        build_request: BuildRequest,
        claim_suffix: str = CLAIM_SUFFIX,
        document_suffixes: tuple[str, ...] = DOCUMENT_SUFFIXES,
    ):
        if build_request is None:
            raise TypeError("必须注入 build_request —— 申请单校验只允许有一处实现")
        self.root = Path(root)
        self.build_request = build_request
        self.claim_suffix = claim_suffix
        self.document_suffixes = tuple(s.lower() for s in document_suffixes)

    # ---- 扫描 ----

    def _pairs(self):
        """产出 ``(document, claim|None)``。排序保证处理顺序可复现。"""
        if not self.root.is_dir():
            return
        for document in sorted(self.root.iterdir()):
            if not document.is_file() or document.name.startswith("."):
                continue
            if document.suffix.lower() not in self.document_suffixes:
                continue
            claim = document.with_name(document.stem + self.claim_suffix)
            yield document, (claim if claim.is_file() else None)

    def _event_id(self, document: Path, claim: Path | None) -> str:
        digest = hashlib.sha256()
        digest.update(document.name.encode("utf-8"))
        digest.update(document.read_bytes())
        if claim is not None:
            digest.update(b"\0")
            digest.update(claim.read_bytes())
        return digest.hexdigest()[:16]

    # ---- 处理 ----

    async def poll_once(self, runtime: AgentRuntime) -> list[RunReport]:
        """扫一轮：能受理的受理，剩下的按结果移走。

        返回**本轮真正新受理的报告**（幂等跳过的、失败的都不在里面）——
        调用方（server 端点 / CLI）据此展示，不用自己再筛。
        """
        handled: list[RunReport] = []
        seen = self.ledger_ids()

        for document, claim in self._pairs():
            event_id = self._event_id(document, claim)

            if event_id in seen:
                # 已经受理过：只是文件又出现在收件口（重启后残留、或人工重复投放）
                self._quarantine(document, claim, event_id, DONE_DIR)
                continue

            if claim is None:
                self._quarantine(
                    document,
                    None,
                    event_id,
                    FAILED_DIR,
                    f"缺少申请单：{document.stem}{self.claim_suffix}（没有申请单就没有申报人，不能替人编一个）",
                )
                continue

            try:
                spec = json.loads(claim.read_text(encoding="utf-8"))
                request = self.build_request(spec)
            except Exception as exc:  # noqa: BLE001 —— 坏申请单是常见情况，不是异常情况
                self._quarantine(document, claim, event_id, FAILED_DIR, f"申请单不可用：{exc}")
                continue

            try:
                report = await runtime.submit(document, request)
            except Exception as exc:  # noqa: BLE001 —— 抽取失败等；不重试，交人工
                self._quarantine(document, claim, event_id, FAILED_DIR, f"受理失败：{exc}")
                continue

            self._record(event_id, document, report.audit_id)
            self._quarantine(document, claim, event_id, DONE_DIR)
            handled.append(report)

        return handled

    # ---- 账本 ----

    def ledger_ids(self) -> set[str]:
        """已受理事件的 id 集合。坏行跳过 —— 账本残缺好过整轮罢工。"""
        path = self.root / LEDGER_NAME
        if not path.is_file():
            return set()
        ids: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("event_id", "")))
            except json.JSONDecodeError:
                continue
        return ids

    def ledger(self) -> list[dict]:
        path = self.root / LEDGER_NAME
        if not path.is_file():
            return []
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def _record(self, event_id: str, document: Path, audit_id: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "event_id": event_id,
            "document": document.name,
            "audit_id": audit_id,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with (self.root / LEDGER_NAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- 移动 ----

    def _quarantine(
        self,
        document: Path,
        claim: Path | None,
        event_id: str,
        bucket: str,
        error: str = "",
    ) -> Path:
        """把一对文件移进 done/ 或 failed/。**只搬不改** —— 原件内容一个字节都没动。"""
        target = self.root / bucket / event_id
        target.mkdir(parents=True, exist_ok=True)
        for path in (document, claim):
            if path is None or not path.is_file():
                continue
            destination = target / path.name
            if destination.exists():  # 同名重复（极端情况）：加序号，不覆盖
                destination = target / f"{path.stem}.{len(list(target.iterdir()))}{path.suffix}"
            path.replace(destination)
        if error:
            (target / "error.txt").write_text(error + "\n", encoding="utf-8")
        return target
