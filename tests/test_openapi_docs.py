"""接口文档：分组、隐藏、边界说明
================================

文档也是产品的一部分。这组测试守三件事：

1. 给外部看的 OpenAPI 里**不该出现返回 HTML 的页面路由**；
2. 每个接口都有分组和业务语言摘要（不再出现 ``default`` 和函数名直译）；
3. 文档必须**如实**写清能力边界 —— 未鉴权、agent 不能签字、哈希链不是可信时间戳、
   agent 没有后台调度器。写了但做不到是撒谎，能做了但没写是不负责任。
"""

from __future__ import annotations

import server


def _schema() -> dict:
    return server.app.openapi()


def test_page_routes_are_hidden_from_schema():
    paths = _schema()["paths"]
    assert "/" not in paths, "根路径（重定向到页面）不该出现在 API 文档里"
    assert "/api/health" not in paths, "健康检查对集成方无意义，隐藏（不是删除）"
    # 注意 /audit 这个路径本身是 ERP 的「无状态审核」接口（POST），
    # 要隐藏的是返回 HTML 的那一个（GET）。
    assert "get" not in paths.get("/audit", {}), "HTML 审核台不该出现在 API 文档里"
    assert "post" in paths["/audit"], "ERP 的无状态审核接口必须留下"


def test_every_route_is_grouped_and_business_named():
    schema = _schema()
    groups = {t["name"] for t in schema.get("tags", [])}
    assert {"核心审核", "Agent 编排", "ERP 集成", "系统"} <= groups
    assert "default" not in groups

    for path, ops in schema["paths"].items():
        for method, op in ops.items():
            assert op.get("tags"), f"{method.upper()} {path} 没有分组"
            assert op.get("summary"), f"{method.upper()} {path} 没有业务摘要"
            # 摘要不该是函数名的直译（如 "Audit Run"）
            assert " " not in op["summary"], f"{method.upper()} {path} 摘要像函数名"


def test_description_states_real_boundaries():
    desc = _schema()["info"]["description"]
    assert "没有鉴权" in desc
    assert "127.0.0.1" in desc
    assert "decide" in desc and "唯一" in desc
    assert "不是可信时间戳" in desc
    assert "没有后台调度器" in desc
    assert "错误码" in desc and "409" in desc


def test_key_endpoints_carry_response_examples():
    """对接方看 schema 只知道类型，看不到长什么样 —— 关键接口必须有示例。"""
    paths = _schema()["paths"]
    for path, method in (
        ("/api/audit/run", "post"),
        ("/api/audit/{audit_id}/decide", "post"),
        ("/api/agent/{audit_id}/timeline", "get"),
        ("/api/agent/scan", "post"),
    ):
        responses = paths[path][method]["responses"]
        assert "200" in responses, f"{path} 缺 200 响应说明"
        example = responses["200"].get("content", {}).get("application/json", {}).get("example")
        assert example, f"{path} 缺 200 响应示例"


def test_409_has_two_known_meanings():
    """409 在本项目有两种含义（已决定 / 扫描中），文档必须分开写。"""
    desc = _schema()["info"]["description"]
    assert "已被决定" in desc and "扫描正在进行" in desc


def test_request_examples_are_provided():
    schemas = _schema()["components"]["schemas"]
    for name in ("AuditRunRequest", "AuditDecideRequest", "ExtractRequest"):
        assert schemas[name].get("example"), f"{name} 缺示例，对接方只能猜"
