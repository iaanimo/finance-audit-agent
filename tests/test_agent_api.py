"""agent 端点：能看、能补动作，**不能替人签字**
================================================

这一组测试盯的是接口的**权力边界**，不是功能：
- 没有"批准"端点（README 里那句话不能只是说说）；
- 时间线能分清 system / agent / human；
- 扫描幂等，重复投放不产生第二张单。
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
from agent.events import ScanLock

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
def temp_data_dir(tmp_path, monkeypatch):
    """把 data/ 指到临时目录 —— 与 test_server.py 同一套做法，绝不写真实目录。"""

    class _FakeSettings:
        def __init__(self) -> None:
            self.project_root = tmp_path
            self.data_dir = tmp_path / "data"
            # 没有 Key：叙述层应自动退回模板，而不是让审核失败
            self.api_key = ""
            self.base_url = ""
            self.model = ""

    monkeypatch.setattr("tools.file_ops.get_settings", lambda: _FakeSettings())
    monkeypatch.setattr("config.settings.get_settings", lambda: _FakeSettings())
    return tmp_path / "data"


@pytest.fixture
def client(temp_data_dir):
    return TestClient(server.app)


def _drop(inbound: Path) -> None:
    inbound.mkdir(parents=True, exist_ok=True)
    shutil.copy(SAMPLES_PDF / "S01_hotel_ok.pdf", inbound / "S01_hotel_ok.pdf")
    (inbound / "S01_hotel_ok.claim.json").write_text(
        json.dumps(CLAIM, ensure_ascii=False), encoding="utf-8"
    )


def _scan(client) -> list[dict]:
    resp = client.post("/api/agent/scan")
    assert resp.status_code == 200
    return resp.json()["handled"]


def test_actions_endpoint_exposes_tiers(client):
    data = client.get("/api/agent/actions").json()
    by_action = {item["action"]: item for item in data["actions"]}

    assert len(by_action) == 9
    for l3 in ("approve", "pay", "file_archive"):
        assert by_action[l3]["tier"] == 3
        assert by_action[l3]["agent_allowed"] is False
        assert by_action[l3]["legal_basis"]  # 拒绝要给依据，不是"系统不允许"
    assert by_action["notify_reviewer"]["agent_allowed"] is True


def test_scan_then_timeline_and_inbox(client, temp_data_dir):
    _drop(temp_data_dir / "inbound")
    handled = _scan(client)
    assert len(handled) == 1
    audit_id = handled[0]["audit_id"]

    timeline = client.get(f"/api/agent/{audit_id}/timeline").json()["timeline"]
    events = [row["event"] for row in timeline]
    assert "agent.run.started" in events
    assert "agent.run.finished" in events
    assert any(row["actor_type"] == "agent" for row in timeline)
    assert any(row["actor_type"] == "system" for row in timeline)

    inbox = client.get("/api/agent/inbox").json()
    assert inbox["who"] == "审核台"
    assert inbox["messages"][0]["audit_id"] == audit_id
    assert inbox["unread"] == 1


def test_scan_is_idempotent(client, temp_data_dir):
    _drop(temp_data_dir / "inbound")
    assert len(_scan(client)) == 1
    # 同一份材料再投放一次：不产生第二张单
    _drop(temp_data_dir / "inbound")
    assert _scan(client) == []
    assert len(client.get("/api/audit/list").json()["audits"]) == 1


def test_timeline_404_for_unknown_id(client):
    assert client.get("/api/agent/nonexistent/timeline").status_code == 404


def test_inbox_mark_read(client, temp_data_dir):
    _drop(temp_data_dir / "inbound")
    _scan(client)
    message_id = client.get("/api/agent/inbox").json()["messages"][0]["id"]

    assert client.post(f"/api/agent/inbox/{message_id}/read").json() == {"ok": True}
    assert client.get("/api/agent/inbox").json()["unread"] == 0
    # 已读再点一次 = 404，不重复记
    assert client.post(f"/api/agent/inbox/{message_id}/read").status_code == 404


def test_inbound_endpoint_lists_pending_without_moving(client, temp_data_dir):
    """收件口状态是**只读**：列得出文件，但不能顺手把文件移走/受理。"""
    inbound = temp_data_dir / "inbound"
    inbound.mkdir(parents=True, exist_ok=True)
    _drop(inbound)

    d = client.get("/api/agent/inbound").json()
    assert d["pending_count"] == 1
    assert d["pending"][0]["document"] == "S01_hotel_ok.pdf"
    assert d["pending"][0]["claim"] == "S01_hotel_ok.claim.json"
    # 只读：文件还在收件口，账本也没动
    assert (inbound / "S01_hotel_ok.pdf").is_file()
    assert d["processed"] == 0


def test_recover_endpoint_reports_nothing_when_clean(client):
    assert client.post("/api/agent/recover").json() == {"recovered": []}


def test_remind_endpoint_is_callable(client):
    """催办是"调用时判断"，端点必须能调；没有超时单时返回空列表。"""
    resp = client.post("/api/agent/remind")
    assert resp.status_code == 200
    assert resp.json() == {"reminded": []}


def test_no_endpoint_can_approve_on_behalf_of_human(client):
    """agent 这一层没有"批准"入口 —— 有的话，前面所有设计都白做。"""
    paths = client.get("/openapi.json").json()["paths"]
    agent_paths = {path: methods for path, methods in paths.items() if path.startswith("/api/agent")}
    assert agent_paths, "agent 路由没挂上"
    for path, methods in agent_paths.items():
        assert "decide" not in path
        for method in methods:
            # 只允许 POST 触发"补动作/扫描/已读"，不允许 PUT/PATCH 之类
            assert method in {"get", "post"}, f"{path} 用了 {method}"


def test_decision_closes_the_inbox_item(client, temp_data_dir):
    """人做完决定，该单的待办必须关闭 —— 而且关闭它的是人，不是 agent。"""
    _drop(temp_data_dir / "inbound")
    audit_id = _scan(client)[0]["audit_id"]
    assert client.get("/api/agent/inbox").json()["open"] == 1

    resp = client.post(
        f"/api/audit/{audit_id}/decide",
        json={"decision": "APPROVED", "operator": "李会计"},
    )
    assert resp.status_code == 200, resp.text

    inbox = client.get("/api/agent/inbox").json()
    assert inbox["open"] == 0
    assert inbox["unread"] == 1  # 没打开过，但已经办完了
    assert inbox["messages"][0]["resolved_by"] == "李会计"

    timeline = client.get(f"/api/agent/{audit_id}/timeline").json()["timeline"]
    row = next(r for r in timeline if r["event"] == "inbox.resolved")
    assert row["actor_type"] == "human"
    assert row["actor"] == "李会计"


def test_scan_returns_409_while_another_scan_runs(client, temp_data_dir):
    inbound = temp_data_dir / "inbound"
    inbound.mkdir(parents=True, exist_ok=True)
    lock = ScanLock(inbound)
    lock.acquire()
    try:
        resp = client.post("/api/agent/scan")
        assert resp.status_code == 409
        assert "正在" in resp.json()["detail"]
    finally:
        lock.release()
    # 锁释放后恢复可用
    assert client.post("/api/agent/scan").status_code == 200


def test_upload_entry_also_records_agent_events(client):
    """上传入口与事件源走同一条受理路径：任意入口进来的单子都有完整轨迹。"""
    pdf = SAMPLES_PDF / "S01_hotel_ok.pdf"
    if not pdf.is_file():
        pytest.skip("样本票缺失")
    resp = client.post(
        "/api/audit/run",
        json={
            "filename": "S01_hotel_ok.pdf",
            "content_b64": base64.b64encode(pdf.read_bytes()).decode(),
            "request": dict(CLAIM),
        },
    )
    assert resp.status_code == 200, resp.text
    audit_id = resp.json()["audit_id"]

    timeline = client.get(f"/api/agent/{audit_id}/timeline").json()["timeline"]
    events = [row["event"] for row in timeline]
    assert "agent.run.started" in events
    assert "agent.run.finished" in events
    assert any(row["actor_type"] == "agent" for row in timeline)

    inbox = client.get("/api/agent/inbox").json()
    assert inbox["messages"][0]["audit_id"] == audit_id


def test_reset_clears_agent_state_but_keeps_archive(client, temp_data_dir, monkeypatch):
    """重置要覆盖 agent 侧运行状态，但**不得**碰已归档的终稿。"""
    _drop(temp_data_dir / "inbound")
    assert len(_scan(client)) == 1

    archive_dir = temp_data_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    kept = archive_dir / "kept.final.json"
    kept.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(server, "DEMO_MODE", True)
    resp = client.post("/api/audit/reset")
    assert resp.status_code == 200
    assert resp.json()["agent_state_removed"] >= 1

    # 事件源账本与收件箱清空
    assert not (temp_data_dir / "inbound" / ".processed.jsonl").exists()
    assert client.get("/api/agent/inbox").json()["messages"] == []
    # 档案保留：它不是运行状态
    assert kept.is_file(), "会计档案不是运行状态，重置无权删除"

    # 重置后同一份材料可以再次受理
    _drop(temp_data_dir / "inbound")
    assert len(_scan(client)) == 1
