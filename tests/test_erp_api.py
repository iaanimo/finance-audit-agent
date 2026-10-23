"""ERP 无状态审核接口（finance/erp_api.py）测试
===============================================

钉住契约四件事：
1. GET /version 必带 rule_version + code_ref；
2. POST /audit 无状态（响应无 audit_id、不碰 AuditStore/run_audit）、零 LLM；
3. severity 三档映射：PASS→info / WARN→warn / FAIL→block / N/A→info / 异常→warn；
4. DTO 问题 → 400 且**响应体仍带 rule_version**（契约：任何响应必带）。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import server
from finance import erp_api
from finance.models import AuditFinding, Severity

client = TestClient(server.app)


def _minimal_doc(**overrides) -> dict:
    doc = {
        "doc_type": "EXPENSE_CLAIM",
        "doc_no": "BX-2026-001",
        "doc_date": "2026-10-23",
        "total_amount": "100.00",
        "currency_code": "CNY",
        "partner": {"name": "测试供应商", "partner_type": "vendor"},
        "lines": [{"material_code": "M0001", "quantity": 2, "price": 50}],
        "extra": {"expense_type": "市内交通费", "reason": "打车"},
    }
    doc.update(overrides)
    return {"dto_version": "1", "document": doc}


def test_version_carries_rule_identity() -> None:
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rule_version"] == erp_api.RULE_VERSION
    assert body["code_ref"] == erp_api.CODE_REF


def test_audit_stateless_no_store_no_llm(monkeypatch) -> None:
    """集成口必须无状态：碰 store / run_audit 即违规（那是 Web UI 管线的形态）。"""
    def _boom(*_a, **_k):  # pragma: no cover - 触发即测试失败
        raise AssertionError("无状态壳不得触碰落盘管线")

    monkeypatch.setattr(server.AUDIT_STORE, "save", _boom, raising=False)
    monkeypatch.setattr("finance.audit.run_audit", _boom)

    resp = client.post("/audit", json=_minimal_doc())
    assert resp.status_code == 200
    body = resp.json()
    assert "audit_id" not in body  # 不建审核单
    assert body["rule_version"] == erp_api.RULE_VERSION
    assert body["code_ref"] == erp_api.CODE_REF
    assert body["checked_at"]
    assert isinstance(body["findings"], list)
    for f in body["findings"]:
        assert set(f) == {"rule_code", "severity", "message", "ref"}
        assert f["severity"] in {"info", "warn", "block"}


def test_severity_mapping_three_tiers() -> None:
    def _finding(**kw) -> AuditFinding:
        base = dict(
            rule_id="R999", clause="1.1", title="t", message="m",
            severity=Severity.PASS,
        )
        base.update(kw)
        return AuditFinding(**base)

    assert erp_api._map_finding(_finding())["severity"] == "info"
    assert erp_api._map_finding(_finding(severity=Severity.WARN))["severity"] == "warn"
    assert erp_api._map_finding(_finding(severity=Severity.FAIL))["severity"] == "block"
    assert erp_api._map_finding(_finding(applicable=False))["severity"] == "info"
    assert erp_api._map_finding(_finding(error="boom"))["severity"] == "warn"


def test_bad_dto_returns_400_with_rule_version() -> None:
    resp = client.post("/audit", json=_minimal_doc(doc_date="2026/10/23"))
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["rule_version"] == erp_api.RULE_VERSION
    assert "doc_date" in detail["message"]


def test_unsupported_dto_version_rejected() -> None:
    resp = client.post("/audit", json=_minimal_doc())
    req = resp.json()
    assert req["rule_version"]  # 正常响应带版本
    bad = {"dto_version": "999", "document": _minimal_doc()["document"]}
    resp = client.post("/audit", json=bad)
    assert resp.status_code == 400
    assert resp.json()["detail"]["rule_version"] == erp_api.RULE_VERSION
