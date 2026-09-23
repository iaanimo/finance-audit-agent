"""提交入口的强制校验测试

审核台的表单可以「清空」，但**清空 ≠ 拦得住**。
实测过：清空后什么都不填直接点提交，请求照发，服务端把申请人当空串、
金额按 0 处理，真生成了一张 R010「申请金额 0.00 ≠ 发票 380.00」的废单。

废单没有财务风险（0 元不可能被放行），但它**真的落进了查重台账** ——
一个宣称「让人没法顺手犯错」的系统，不该留这个口子。

两道拦截各测各的：

- 页面那道（``static/audit.html`` 的 ``describeFormProblem``）拦的是手滑，
  代价是用户看不到即时反馈之前先发一次请求；
- 服务端那道（``server.py::_build_request``）拦的是**绕过页面直接 POST**，
  这才是拦得住的那一道。

两处的必填清单必须一致 —— :func:`test_page_and_server_agree_on_required_fields`
就是盯着这件事的。
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
from finance import parse_money
from server import _build_request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_HTML = PROJECT_ROOT / "static" / "audit.html"

#: 一份合法申请单，各测试按需覆写其中一项
VALID_FORM = {
    "applicant": "张三",
    "department": "技术部",
    "expense_type": "住宿费",
    "amount": "1650.00",
    "reason": "上海客户现场支持",
    "submit_date": "2026-09-18",
    "city": "上海",
    "nights": "3",
}

#: 页面上的申请单字段（与 audit.html 的表单一致）
FORM_FIELDS = [
    "applicant",
    "department",
    "expense_type",
    "amount",
    "reason",
    "submit_date",
    "city",
    "nights",
    "headcount",
]


@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch):
    """把 ``resolve_data_path`` 指到临时目录，测试绝不写进真实 data/。"""
    root = tmp_path

    class _FakeSettings:
        def __init__(self) -> None:
            self.project_root = root
            self.data_dir = root / "data"

    monkeypatch.setattr("tools.file_ops.get_settings", lambda: _FakeSettings())
    return root / "data"


@pytest.fixture
def client(temp_data_dir):
    return TestClient(server.app)


def _server_rejects_blank(field: str) -> bool:
    """把某个字段置空，看服务端是否拒绝。"""
    spec = dict(VALID_FORM)
    spec[field] = ""
    try:
        _build_request(spec)
    except (KeyError, ValueError):
        return True
    return False


# ---------------------------------------------------------------------------
# 必填项
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, label",
    [
        ("applicant", "申请人"),
        ("department", "所属部门"),
        ("expense_type", "费用类型"),
    ],
)
def test_blank_text_field_is_rejected(field, label):
    """全空格也算空 —— ``strip()`` 之后才判断。"""
    spec = dict(VALID_FORM, **{field: "   "})
    with pytest.raises(ValueError) as exc:
        _build_request(spec)
    assert label in str(exc.value)


def test_all_blank_fields_are_listed_at_once():
    """一次报全，别让人填一个报一个。"""
    with pytest.raises(ValueError) as exc:
        _build_request({"submit_date": "2026-09-18", "amount": "1", "nights": ""})
    msg = str(exc.value)
    assert "申请人" in msg and "所属部门" in msg and "费用类型" in msg


def test_json_null_is_not_the_string_none():
    """前端传 ``null`` 时，``str(None)`` 会变成字面量 ``"None"`` —— 那是个坑。"""
    with pytest.raises(ValueError):
        _build_request(dict(VALID_FORM, applicant=None))


@pytest.mark.parametrize("amount", ["0", "0.00", "", None, "-1.00"])
def test_non_positive_amount_is_rejected(amount):
    """金额 0 不是「小金额」，是**没填** —— 放过去只会得到一张废单。"""
    with pytest.raises(ValueError) as exc:
        _build_request(dict(VALID_FORM, amount=amount))
    assert "金额" in str(exc.value)


def test_amount_noise_is_tolerated():
    """照票面抄下来的 ``1,650.00`` / ``¥1650`` 也算合法输入。"""
    for raw in ("1,650.00", "¥1650", " 1650 元"):
        assert _build_request(dict(VALID_FORM, amount=raw)).amount == parse_money(raw)


def test_submit_date_is_required_not_defaulted():
    """提交日期没有默认值。

    曾经缺省取服务器当天 —— 那等于系统替申请人编了一个申报日期，
    而 R003（开票日距提交日 <= 60 天、不得跨年）正是拿它当判定基准的。
    """
    with pytest.raises(ValueError) as exc:
        _build_request(dict(VALID_FORM, submit_date=""))
    assert "提交日期" in str(exc.value)


def test_malformed_submit_date_is_rejected():
    with pytest.raises(ValueError) as exc:
        _build_request(dict(VALID_FORM, submit_date="2026/09/18"))
    assert "YYYY-MM-DD" in str(exc.value)


def test_valid_form_still_builds():
    """拦得住的另一面：合法申请单必须照常通过，别把拦截面做宽了。"""
    req = _build_request(dict(VALID_FORM))
    assert req.applicant == "张三"
    assert req.department == "技术部"
    assert req.expense_type == "住宿费"
    assert req.amount == parse_money("1650.00")
    assert req.nights == 3
    assert req.headcount is None          # 未填就是 None，不是 0


# ---------------------------------------------------------------------------
# 两处必填清单必须一致（页面 ↔ 服务端）
# ---------------------------------------------------------------------------


def _page_required_field_ids() -> list[str]:
    block = re.search(r"var REQUIRED_FIELDS = \[(.*?)\];", AUDIT_HTML.read_text(encoding="utf-8"), re.S)
    assert block, "static/audit.html 里找不到 REQUIRED_FIELDS 清单"
    return re.findall(r"\['([a-z_]+)'", block.group(1))


def test_page_and_server_agree_on_required_fields():
    """页面必填清单 ↔ 服务端必填校验，两处必须一致。

    只改一边的后果很具体：
    页面不拦、服务端拦 → 用户看到的是迟到的 400；
    页面拦、服务端不拦 → 直接 POST 就能生成废单。
    所以新增一个必填项时，两边都得改 —— 这个测试就是盯着这件事的。
    """
    page_required = set(_page_required_field_ids())
    server_required = {f for f in FORM_FIELDS if _server_rejects_blank(f)}

    # 部门是例外：页面上它是个下拉框，产不出空值，所以不必进页面清单。
    assert page_required | {"department"} == server_required


# ---------------------------------------------------------------------------
# 端到端：空表单必须 400，且一个字节都不落盘
# ---------------------------------------------------------------------------


def test_blank_form_is_rejected_before_writing_anything(client, temp_data_dir):
    """校验必须发生在落盘**之前**。

    反过来写的话，一张缺字段的废单会先把文件写进 data/uploads/ 再被 400 拒掉，
    留下一个没人引用的孤儿文件。
    """
    resp = client.post(
        "/api/audit/run",
        json={
            "filename": "x.pdf",
            "content_b64": base64.b64encode(b"%PDF-1.4\n%%EOF").decode(),
            "request": {},
        },
    )
    assert resp.status_code == 400
    assert "报销申请单" in resp.json()["detail"]
    assert not (temp_data_dir / "uploads").exists(), "被拒的请求不该留下任何文件"


def test_blank_form_via_http_lists_the_missing_fields(client, temp_data_dir):
    resp = client.post(
        "/api/audit/run",
        json={
            "filename": "x.pdf",
            "content_b64": base64.b64encode(b"%PDF-1.4\n%%EOF").decode(),
            "request": {"submit_date": "2026-09-18"},
        },
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "申请人" in detail and "费用类型" in detail
    assert not (temp_data_dir / "uploads").exists()


# ---------------------------------------------------------------------------
# 「清空演示数据」默认关闭 —— 这是法定义务，不是产品定位选择
# ---------------------------------------------------------------------------


def test_reset_is_disabled_by_default(client):
    """默认启动时，删除会计档案的接口必须是 403。

    依据：《会计档案管理办法》（财政部、国家档案局令第 79 号）第十四条、第十五条
    及附表 —— 原始凭证、记账凭证的最低保管期限为 30 年；本项目制度 6.2 也写着
    「留痕记录只追加，不得修改或删除」。

    真实系统里正确的更正方式是**红冲**（生成反向凭证），不是删除。
    这条测试守的是"别哪天为了演示方便，把这个口子又默认打开"。
    """
    assert server.DEMO_MODE is False, "测试进程必须是默认（非演示）状态"

    resp = client.post("/api/audit/reset")
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert "不得删除" in detail
    assert "30 年" in detail          # 把法律依据写在报错里，而不是只说"没权限"
    assert "红冲" in detail


def test_reset_opens_only_in_demo_mode(client, monkeypatch):
    """显式开演示模式时才放行 —— 演示流程不受影响（start.bat 会带 --demo）。"""
    monkeypatch.setattr(server, "DEMO_MODE", True)
    resp = client.post("/api/audit/reset")
    assert resp.status_code == 200


def test_health_reports_demo_mode(client, monkeypatch):
    """页面靠这个字段决定显不显示清空按钮，别让它失联。"""
    assert client.get("/api/health").json()["demo_mode"] is False
    monkeypatch.setattr(server, "DEMO_MODE", True)
    assert client.get("/api/health").json()["demo_mode"] is True


# ---------------------------------------------------------------------------
# 非法 audit_id 必须是 404，不是 500
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/api/audit/!!!", "/api/audit/!!!/log"])
def test_illegal_audit_id_is_404_not_500(client, path):
    """用户输入不该打出未捕获异常。

    实测过：`/api/audit/!!!` 会让 ``store._safe_id`` 抛 ValueError，
    而三个端点都没接 —— 直接 500，堆栈进日志。路径穿越本身是被挡住的
    （``_safe_id`` 会拒），但"挡住了"和"体面地拒绝"是两回事。
    """
    assert client.get(path).status_code == 404


def test_illegal_audit_id_on_decide_is_404(client):
    resp = client.post(
        "/api/audit/!!!/decide", json={"decision": "APPROVED", "operator": "x"}
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 页面不许写死判定结论
# ---------------------------------------------------------------------------


def test_page_reads_balance_verdict_from_backend():
    """凭证的「借贷平衡」必须读后端的 ``v.balanced``，不能写死。

    R013 现在是真算术校验（借 = 不含税 + 进项税，贷 = 价税合计），
    系统**真的可能产出不平衡的凭证**。页面写死「借贷平衡」的话，
    一张不平衡的凭证在界面上照样显示绿标签 —— 那是页面在替系统撒谎。

    这是静态检查（没有浏览器可跑）。它拦不住"改坏了渲染逻辑"，
    但拦得住"改回写死"这一个具体动作。
    """
    html = AUDIT_HTML.read_text(encoding="utf-8")
    assert "v.balanced" in html, "页面没有读后端的借贷平衡结论"
    assert "借贷不平衡" in html, "页面没有不平衡时的分支"
    assert html.count("借贷平衡") == 1, (
        "「借贷平衡」出现了不止一次 —— 怀疑又有一处写死的标签"
    )


# ---------------------------------------------------------------------------
# import 不许有副作用
# ---------------------------------------------------------------------------


def test_importing_server_does_not_attach_file_handler():
    """``import server`` 不许碰真实文件 —— 日志初始化只在 ``main()`` 里做。

    回归背景：server.py 曾在 import 时 ``LOGS_DIR.mkdir`` + ``logging.FileHandler``，
    测试一 import 就往真实 logs/ 里写；在只读环境（只读挂载、CI 沙箱）里更是
    直接 PermissionError，整个测试收集都会中断（实测踩过一次，129 条测试
    一条都没跑起来）。日志是运行期的事，就该在启动时做。
    """
    import logging as _logging

    # 只看"挂在 logs/ 里的文件 handler" —— pytest 自己可能往 NUL 挂 handler，
    # 那不是 import 副作用；真正的回归是 server 模块在 import 时打开真实日志文件
    offenders = [
        h
        for h in _logging.getLogger().handlers
        if isinstance(h, _logging.FileHandler)
        and str(getattr(h, "baseFilename", "")).startswith(str(server.LOGS_DIR))
    ]
    assert not offenders, f"root logger 挂了 logs/ 里的文件 handler（import 副作用）：{offenders}"


# ---------------------------------------------------------------------------
# 按票面自动填：/api/audit/extract 的回填边界
# ---------------------------------------------------------------------------

SAMPLES_PDF = PROJECT_ROOT / "finance" / "samples" / "pdf"


def test_extract_endpoint_prefills_only_invoice_facts(client, temp_data_dir):
    """/api/audit/extract 只回"票面上有的"：金额、票面日期、费用类型**推荐**。

    申报信息（申请人/事由/提交日期/人数/晚数）**一律不得出现在响应里** ——
    给了默认值就等于替申请人编申报数据（同「提交日期不设默认值」的理由）。
    白名单断言锁死返回字段集，多加一个字段都会在这里被拦下问一句"这是票面有的吗"。
    """
    pdf = SAMPLES_PDF / "S01_hotel_ok.pdf"
    if not pdf.is_file():
        pytest.skip("样本票缺失")
    resp = client.post(
        "/api/audit/extract",
        json={
            "filename": "S01_hotel_ok.pdf",
            "content_b64": base64.b64encode(pdf.read_bytes()).decode(),
        },
    )
    assert resp.status_code == 200
    d = resp.json()
    assert d["invoice"]["total"] == 1650.00
    assert d["invoice"]["issue_date"] is not None      # 票面日期：票面有的
    assert d["suggest"]["expense_type"] == "住宿费"     # 推荐，不是替人决定
    # 票面事实全量返回 + 每字段置信度 + 关键字段清单（票面确认卡的数据源）
    assert set(d["invoice"].keys()) == {
        "invoice_code", "invoice_number", "invoice_type", "issue_date",
        "buyer_name", "buyer_tax_id", "seller_name", "seller_tax_id",
        "item_name", "amount", "tax_rate", "tax_amount", "total",
        "total_in_words", "remark",
    }
    assert set(d["confidence"].keys()) == set(d["invoice"].keys())
    assert d["confidence"]["total"] == "high"          # PDF 文本层结构化抽取
    assert set(d["critical_fields"]) == {"total", "buyer_tax_id", "invoice_number"}
    assert set(d.keys()) == {"invoice", "confidence", "critical_fields", "suggest"}
    # 申报信息（申请人/事由/提交日期/人数/晚数）不得出现在返回里 —— 票面没有，不许替人编
    # 抽取不建单、不落盘（临时文件即用即删）
    assert not (temp_data_dir / "uploads").exists()


def test_extract_endpoint_reports_failure_as_422_not_500(client):
    """抽不出来是可预期的（版式不支持/文件损坏）——422 人话，不是 500。"""
    resp = client.post(
        "/api/audit/extract",
        json={"filename": "x.pdf", "content_b64": base64.b64encode(b"not a pdf").decode()},
    )
    assert resp.status_code == 422
    assert "手工填写" in resp.json()["detail"]


def test_critical_field_change_requires_original_declaration(client):
    """改关键字段（金额/税号/号码）必须带「已逐项核对原件」声明 —— 服务端强拦。

    服务端自己核不了原件，但**可以让人工留下可追责的声明**：改可以改，
    改了要对原件负责。声明缺失 -> 400，且**落盘之前**就拦（不留孤儿文件）。
    """
    pdf = SAMPLES_PDF / "S01_hotel_ok.pdf"
    if not pdf.is_file():
        pytest.skip("样本票缺失")
    body = {
        "filename": "S01_hotel_ok.pdf",
        "content_b64": base64.b64encode(pdf.read_bytes()).decode(),
        "request": dict(VALID_FORM),
        "invoice_overrides": {"total": "1650.01"},
        "field_changes": [{"field": "total", "from": "1650.0", "to": "1650.01"}],
        "critical_confirmed": False,
    }
    resp = client.post("/api/audit/run", json=body)
    assert resp.status_code == 400
    assert "核对原件" in resp.json()["detail"]
