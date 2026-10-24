"""财务报销审核受控 Agent —— Web 服务
=======================================

启动::

    ./.venv/Scripts/python.exe server.py
    # 浏览器打开 http://127.0.0.1:8100/audit

**这条链路刻意不经过任何 agent 循环，也不给模型任何工具。**

审核走的是一条**固定的、可复现的管线**（``finance/audit.py::run_audit``）：
LLM 只在这条管线的两端出现 —— 入口的字段抽取、出口的叙述润色；
中间 19 条规则的判定完全由 Python 完成。

这不是"没用上 agent 能力"，这是**设计**：财务审核的结论必须可复现，
不能交给一个会自由发挥的循环。

`agent/` 包是**另一层**，且只在外围：它调用 ``run_audit`` 之后决定"通知谁、要什么材料"，
并把每一步写进审计日志。它拿不到批准、付款、归档的钥匙（见 ``agent/gate.py`` 的 L0–L3），
所以"有 agent"和"判定可复现"这两件事并不冲突。

**一个实测过的坑**：本机（以及绝大多数精简部署）没有装 ``python-multipart``，
所以 **不能用** ``UploadFile`` / ``File()`` / ``Form()`` —— 一写就 500。
发票文件走 base64 塞进 JSON 传输。
"""

from __future__ import annotations

import argparse
import base64
import binascii
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from agent import Action, Gate, Inbox
from agent.events import DirectorySource, ScanInProgress
from agent.runtime import REVIEW_DESK, AgentRuntime
from config.settings import get_settings
from finance import AuditResult, Decision, ReimbursementRequest, parse_money
from finance.audit import AuditError, OverrideReasonRequired, decide
from finance.advice import suggest_advice
from finance.erp_api import router as _erp_router
from finance.policy import load_policy_bundle
from finance.store import AuditStore
from tools.file_ops import resolve_data_path

HERE = Path(__file__).resolve().parent
LOGS_DIR = HERE / "logs"
logger = logging.getLogger("finance-audit")


def setup_logging() -> None:
    """初始化日志（建 logs/ 目录 + 挂文件/控制台 handler）。**只在启动时调用。**

    这里刻意**不在 import 时执行**：模块一被导入就建目录、打开日志文件，是
    典型的 import 副作用 —— 测试一 import server 就往真实 logs/ 里写；在只读
    环境（只读容器挂载、CI 沙箱）里 import server 更是直接 PermissionError，
    连测试收集都会整个中断。日志是运行期的事，就该在启动时做。
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "finance-audit.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

#: 应用版本（发布用）。规则版本在 ``finance/erp_api.py::RULE_VERSION``——
#: 两者不同步：应用可以改界面、加端点而不动规则；规则一改就必须升 RULE_VERSION，
#: 否则历史结论没法按当时版本复算。
APP_VERSION = "1.0.0"

API_DESCRIPTION = """
财务报销审核受控 Agent —— 接口文档。

## 这个接口能做什么、不能做什么

- **判定零 LLM**：19 条规则由 Python 逐条算出，结论可复现；每个响应都带 `rule_version`，供调用方事后按版本复算。
- `/api/audit/{audit_id}/decide` 是**唯一**能批准报销的入口，必须由人调用并填写操作人。
- Agent 层只能把单据推进到「待签批」：它可以通知、要材料、补发消息，**不能**批准、付款、归档。
- 审计日志是只追加的哈希链：**改了能被发现**（见 `GET /api/audit/{id}/log`），但它不是可信时间戳，也不是法律级电子签名。

## 鉴权

**当前版本没有鉴权**（单机本地运行，默认只监听 127.0.0.1）。
请勿直接暴露公网；生产接入需补 API Key / SSO —— 接口已留，尚未实现。

## 接入约定

- 金额一律用字符串传输（如 `"1650.00"`），避免二进制浮点误差。
- 错误用 HTTP 状态码 + `{"detail": "..."}`，不套统一 envelope。
- 事件源按「文档内容 + 申请单内容」哈希去重：同一份材料重复投放不会生成第二张单。

## 错误码

| 状态码 | 含义 | 典型 detail |
|---|---|---|
| 400 | 入参不合法：DTO、必填项、关键字段未声明、推翻未写理由 | `报销申请单字段有误：…` |
| 403 | 「清空演示数据」未在 --demo 下开启 | `「清空演示数据」已停用…` |
| 404 | 审核单或消息不存在 | `审核单不存在。` |
| 409 | 冲突，**两种含义**：该单已被决定 / 扫描正在进行 | `已由 李会计 做过决定`、`正在被另一轮扫描处理` |
| 422 | 发票处理失败（抽取失败、文件损坏） | `发票处理失败：…` |

## Agent 是怎么被唤醒的

当前只有**目录轮询**这一条事件源：把「发票 + 申请单」成对放进 `data/inbound/`，
调用 `POST /api/agent/scan` 扫一轮。**没有后台调度器**，也没有 webhook ——
催办不是定时任务：调用 `POST /api/agent/remind` 时才判断哪些单超时，
因此本版本不会自己"醒来"。
"""

app = FastAPI(
    title="财务报销审核受控 Agent",
    description=API_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    openapi_tags=[
        {"name": "核心审核", "description": "提交、查看、人工决定。判定零 LLM，结论可复现。"},
        {"name": "Agent 编排", "description": "事件源扫描、断点恢复、时间线、收件箱。Agent 只能推进到待签批，不能签字。"},
        {"name": "集成接口", "description": "无状态审核：DTO 进、结论出，不落本地 store；对接方自行组装 DTO。"},
        {"name": "系统", "description": "健康检查与演示数据维护。清空仅在 --demo 下可用。"},
    ],
)

# 无状态集成接口：POST /audit + GET /version
# ——DTO 进结论出、不落 store、零 LLM；输入输出形状见 finance/erp_api.py
app.include_router(_erp_router)

AUDIT_STORE = AuditStore()
SAMPLE_DIR = HERE / "finance" / "samples"
UPLOAD_DIR_NAME = "uploads"
#: 事件源收件口：``data/inbound/`` 下"文档 + 申请单"成对出现即被受理。
AGENT_INBOUND_DIR_NAME = "inbound"

#: 演示模式。**只有显式 `--demo` 启动时才为 True**，唯一作用是开放「清空演示数据」。
#:
#: 为什么默认关：删除会计凭证与审计轨迹在真实系统里是**违法**的 ——
#: 《会计档案管理办法》（财政部、国家档案局令第 79 号）第十四条、第十五条及附表
#: 规定「原始凭证、记账凭证」的最低保管期限为 30 年，且第十五条明确附表所列为
#: **最低**期限。本项目自己的制度 6.2 也写着「留痕记录只追加，不得修改或删除」。
#: 真实系统里正确的更正方式是**红冲**（生成一张反向凭证），不是删除。
#:
#: 所以这不是"产品定位选择"，是法定义务 —— 默认必须关，演示时才临时开。
DEMO_MODE = False
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024

_ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".xml", ".ofd"}


# ---------------------------------------------------------------------------
# 接口文档示例（P1）：响应体按**真实跑出来的结构**写，不手编字段
# ---------------------------------------------------------------------------
# 为什么值得写：对接方看 schema 只知道类型，不知道"长什么样"。
# 示例从一次真实 S01 运行里截取（findings 取前两条），字段名与顺序都是真的。

_EX_RUN_OK = {
    "audit_id": "54a58dabd7d6",
    "state": "pending_review",
    "suggested_status": "APPROVED",
    "narrative": "共执行 19 条规则：通过 14 条，不适用 5 条（票种/费用类型不涉及）……",
    "narrative_source": "template",
    "findings": [
        {
            "rule_id": "R001",
            "clause": "3.1",
            "title": "发票抬头必须为公司全称",
            "clause_text": "发票「购买方名称」必须为 示例科技有限公司 全称。……",
            "severity": "PASS",
            "message": "抬头为「示例科技有限公司」，与公司全称一致",
            "evidence": {
                "field": "购买方名称",
                "actual": "示例科技有限公司",
                "expected": "示例科技有限公司",
            },
            "applicable": True,
            "error": None,
        },
        {
            "rule_id": "R006",
            "clause": "4.3",
            "title": "餐饮费人均不超过 150 元",
            "severity": "FAIL",
            "message": "人均 175.00 元（700.00 / 4 人），超过人均限额 150.00 元",
            "evidence": {"per_head": "175.00", "limit": "150.00"},
            "applicable": True,
        },
    ],
    "voucher": {
        "lines": [
            {"direction": "借", "account": "管理费用-差旅费-住宿费", "amount": 1650.0},
            {"direction": "贷", "account": "其他应付款-员工报销", "amount": 1650.0},
        ],
        "summary": "张三 住宿费报销 上海客户现场支持",
        "debit_total": 1650.0,
        "credit_total": 1650.0,
        "balanced": True,
    },
    "decision": None,
    "overridden": False,
}

_EX_TIMELINE = {
    "audit_id": "54a58dabd7d6",
    "timeline": [
        {"seq": 1, "event": "extracted", "actor_type": "system",
         "ts": "2026-10-24T13:40:00+00:00"},
        {"seq": 7, "event": "agent.run.started", "actor": "nono-agent",
         "actor_type": "agent", "ts": "2026-10-24T13:40:00+00:00"},
        {"seq": 8, "event": "agent.action", "action": "notify_reviewer",
         "kind": "draft_ready", "result": "sent", "actor_type": "agent",
         "ts": "2026-10-24T13:40:00+00:00"},
        {"seq": 10, "event": "decided", "decision": "APPROVED",
         "operator": "李会计", "actor_type": "human",
         "ts": "2026-10-24T13:41:12+00:00"},
    ],
}

_RUN_RESPONSES = {
    200: {
        "description": "审核完成：停在「待签批」，等待人工决定（系统不会自己批准）",
        "content": {"application/json": {"example": _EX_RUN_OK}},
    },
    400: {
        "description": "报销申请单字段有误 / 关键字段修改未声明",
        "content": {"application/json": {"example": {"detail": "报销申请单字段有误：申请人未填"}}},
    },
    422: {
        "description": "发票处理失败（抽取失败、文件损坏、不支持的类型）",
        "content": {"application/json": {"example": {"detail": "发票处理失败：无法读取 PDF 文本层"}}},
    },
}

_DECIDE_RESPONSES = {
    200: {
        "description": "决定已落盘（轨迹只追加；推翻系统建议时 override_reason 随单留档）",
        "content": {
            "application/json": {
                "example": {
                    "audit_id": "54a58dabd7d6",
                    "state": "approved",
                    "decision": "APPROVED",
                    "operator": "李会计",
                    "override_reason": "",
                    "overridden": False,
                    "decided_at": "2026-10-24T13:41:12+00:00",
                    "suggested_status": "APPROVED",
                }
            }
        },
    },
    400: {
        "description": "推翻系统建议但未写书面理由（制度 2.3 的服务端强制）",
        "content": {
            "application/json": {
                "example": {
                    "detail": "系统建议为 REJECTED，你的决定是 APPROVED，"
                    "属于推翻系统建议，必须填写书面理由"
                }
            }
        },
    },
    409: {
        "description": "该单已被做过决定（重复点击 / 并发）",
        "content": {
            "application/json": {
                "example": {"detail": "审核单 54a58dabd7d6 已由 李会计 做过决定"}
            }
        },
    },
}

_TIMELINE_RESPONSES = {
    200: {
        "description": "按 seq 排序的轨迹；actor_type 区分 system / agent / human",
        "content": {"application/json": {"example": _EX_TIMELINE}},
    },
    404: {
        "description": "审核单不存在",
        "content": {"application/json": {"example": {"detail": "审核单不存在。"}}},
    },
}

_SCAN_RESPONSES = {
    200: {
        "description": "本轮**真正新受理**的单据（幂等跳过与失败的不计入）",
        "content": {
            "application/json": {
                "example": {
                    "handled": [
                        {
                            "audit_id": "54a58dabd7d6",
                            "state": "pending_review",
                            "suggested_status": "APPROVED",
                            "actions": ["notify_reviewer"],
                            "steps": [
                                {"action": "notify_reviewer", "outcome": "sent", "to": "审核台"}
                            ],
                            "recovered": False,
                        }
                    ],
                    "inbound": "D:\\nono\\data\\inbound",
                }
            }
        },
    },
    409: {
        "description": "另一轮扫描正在进行（不排队，稍后重试）",
        "content": {
            "application/json": {
                "example": {"detail": "data/inbound 正在被另一轮扫描处理，稍后重试"}
            }
        },
    },
}


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class AuditRunRequest(BaseModel):
    filename: str = Field(default="", description="原始文件名，用于留痕与后缀判断")
    content_b64: str = Field(default="", description="发票文件的 base64 内容")
    use_vision: bool = Field(
        default=False, description="PDF 无文本层时是否允许调视觉模型兜底"
    )
    request: dict = Field(default_factory=dict, description="报销申请单字段")
    # ---- AI 预填 + 人工确认（票面确认卡）----
    invoice_overrides: dict = Field(
        default_factory=dict, description="人工确认后的票面字段（覆盖抽取结果）"
    )
    field_changes: list = Field(
        default_factory=list,
        description="字段级修改留痕：[{field, from, to}]，逐笔进审计日志",
    )
    critical_confirmed: bool = Field(
        default=False,
        description="关键字段（金额/税号/发票号码）已逐项核对原件的声明",
    )
    confirmed_by: str = Field(default="", description="确认人（留痕）")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "S01_hotel_ok.pdf",
                "content_b64": "JVBERi0xLjQK（PDF 文件的 base64，此处省略）",
                "request": {
                    "applicant": "张三",
                    "department": "技术部",
                    "expense_type": "住宿费",
                    "amount": "1650.00",
                    "reason": "上海客户现场支持",
                    "submit_date": "2026-09-18",
                    "city": "上海",
                    "nights": 3,
                },
            }
        }
    )


class AuditDecideRequest(BaseModel):
    decision: str = Field(description="APPROVED 或 REJECTED")
    operator: str = Field(default="", description="操作人")
    override_reason: str = Field(
        default="", description="推翻系统建议时的书面理由（制度 2.3 要求必填）"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "decision": "APPROVED",
                "operator": "李会计",
                "override_reason": "客户临时改期，超标部分员工自付，已电话确认",
            }
        }
    )


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def index():
    """根路径直接进审核台。"""
    return RedirectResponse(url="/audit")


@app.get("/audit", include_in_schema=False)
async def audit_page():
    page = HERE / "static" / "audit.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="audit.html 不存在")
    return FileResponse(page)


@app.get("/api/health", include_in_schema=False, tags=["系统"], summary="健康检查（含演示模式标志）")
async def health():
    """健康检查。顺带告诉页面当前是不是演示模式 —— 页面据此决定显不显示清空按钮。"""
    return {"status": "ok", "demo_mode": DEMO_MODE, "version": APP_VERSION}


# ---------------------------------------------------------------------------
# 样本票（演示用）
# ---------------------------------------------------------------------------


@app.get("/api/audit/samples", tags=["核心审核"], summary="列出内置样本票")
async def audit_samples():
    """列出内置合成样本票，供演示时一键选用。"""
    manifest_path = SAMPLE_DIR / "samples.yaml"
    if not manifest_path.is_file():
        return {"samples": []}
    import yaml

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return {
        "samples": [
            {
                "key": key,
                "title": spec.get("title", ""),
                "expects": spec.get("expects", ""),
                "pdf": spec.get("pdf", ""),
                "request": spec.get("request", {}),
            }
            for key, spec in manifest.items()
        ]
    }


@app.get("/api/audit/sample/{key}", tags=["核心审核"], summary="取一张样本票原文")
async def audit_sample_file(key: str):
    """取一张样本票的原文（前端转 base64 后回传，模拟"上传"）。

    **按清单取文件**，不再硬编码 ``pdf/{key}.pdf`` —— 那个写死曾让 xml/ 票据
    样本在界面上"选了必炸"：评测直读文件全绿、界面走本接口 404，两套口径打架。
    （消费方清单纪律：样本的消费方 = 评测（直读）+ 本接口（界面），同一路径源。）
    """
    safe = "".join(ch for ch in key if ch.isalnum() or ch in "-_")
    spec: dict = {}
    manifest_path = SAMPLE_DIR / "samples.yaml"
    if manifest_path.is_file():
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        spec = manifest.get(safe) or {}
    rel = str(spec.get("pdf") or "")
    path = (SAMPLE_DIR / rel).resolve() if rel else (SAMPLE_DIR / "pdf" / f"{safe}.pdf").resolve()
    # 防目录穿越：清单被改坏/塞进 ../ 时也跳不出 samples/（双保险）
    # ⚠️ 原写法是 `str(path).startswith(str(SAMPLE_DIR) + "\\")` —— 那个 "\\" 是
    # **Windows 分隔符**。Linux 上 str(path) 用 "/"，startswith 恒为 False，于是短路失效、
    # 退化到第二个条件 `path.parent != samples/`，把 pdf/、xml/ 子目录下的样本全判 404。
    # 症状很有欺骗性：windows CI 绿、ubuntu CI 红，本地（Windows）永远复现不出来。
    # 改用 is_relative_to()：与分隔符无关，两个平台同一语义。
    if not path.is_relative_to(SAMPLE_DIR.resolve()):
        raise HTTPException(status_code=404, detail=f"样本不存在：{key}")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"样本不存在：{key}")
    return FileResponse(path, filename=path.name)


# ---------------------------------------------------------------------------
# 审核
# ---------------------------------------------------------------------------


@app.get("/api/audit/list", tags=["核心审核"], summary="审核单列表")
async def audit_list():
    return {"audits": AUDIT_STORE.list_audits()}


@app.post(
    "/api/audit/run",
    tags=["核心审核"],
    summary="提交单据审核",
    responses=_RUN_RESPONSES,
)
async def audit_run(req: AuditRunRequest):
    """提交一张发票跑审核。

    **系统只能把单子推到"待人工决定"。** 这个端点里没有任何一行能批准报销。
    """
    if not req.content_b64:
        raise HTTPException(status_code=400, detail="发票内容为空。")

    try:
        blob = base64.b64decode(req.content_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="发票内容不是合法的 base64。")

    if len(blob) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="发票文件过大（上限 8MB）。")

    suffix = Path(req.filename).suffix.lower() or ".pdf"
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix}")

    # ⚠️ 顺序要紧：**先校验，再落盘**。
    # 反过来的话，一张缺字段的废单会先把文件写进 data/uploads/ 再被 400 拒掉，
    # 留下一个没人引用的孤儿文件。

    # 关键字段（金额/税号/发票号码/**发票类型**）被人工改过的，必须带"已逐项核对原件"
    # 声明。服务端自己核不了原件，但**可以让人工留下可追责的声明** —— 留痕精神：
    # 改可以改，改了要有人对原件负责。字段名小写规范化再比（大小写变形不许绕过），
    # 且**以 invoice_overrides 实际内容为准**（B2：不看客户端自报的 field_changes）。
    from finance.extractor import CRITICAL_FIELDS
    from finance.models import Invoice as _InvoiceModel

    _known = {name.lower() for name in _InvoiceModel.model_fields}
    _touched = {str(k).strip().lower() for k in req.invoice_overrides}
    unknown = sorted(_touched - _known)
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"invoice_overrides 含未知票面字段：{unknown}"
        )
    # 证据链元数据（来源文件/原文/抽取方式）**不接受申报口覆盖** ——
    # extraction_method 改一下就能伪造置信度等级（B2 残留教训）。
    # 它们记录"数据从哪来"，要更正只能走抽取层并留痕。
    _protected = {"source_file", "raw_text", "extraction_method"}
    hit = _touched & _protected
    if hit:
        raise HTTPException(
            status_code=400,
            detail=f"证据链元数据（{sorted(hit)}）不可由申报口覆盖 —— 记录的是数据来源，更正须走抽取层。",
        )
    if (_touched & set(CRITICAL_FIELDS)) and not req.critical_confirmed:
        raise HTTPException(
            status_code=400,
            detail="修改了关键字段（金额/税号/发票号码/发票类型），必须勾选「已逐项核对原件」——改可以改，改了要对原件负责并留痕。",
        )

    try:
        request = _build_request(req.request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"报销申请单字段有误：{exc}")

    target = resolve_data_path(f"{UPLOAD_DIR_NAME}/{uuid.uuid4().hex[:10]}{suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)

    # 上传入口与事件源走**同一条受理路径**（runtime.submit → run_audit）。
    # 否则同一张票从不同入口进来，轨迹不一样：一条有 agent 事件，一条没有 ——
    # 而"每一步谁触发的"正是这一层要回答的问题。
    try:
        report = await _agent_runtime().submit(
            target,
            request,
            narrative_llm="auto",  # 与改造前一致：有 Key 就润色，没 Key 自动退回模板
            use_vision=req.use_vision,
            invoice_overrides=req.invoice_overrides,
            field_changes=req.field_changes,
            confirmed_by=req.confirmed_by or request.applicant,
        )
    except Exception as exc:  # noqa: BLE001 —— 抽取失败给人话提示，不是 500
        logger.warning("audit run failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"发票处理失败：{exc}")

    result = AUDIT_STORE.load(report.audit_id)
    if result is None:
        raise HTTPException(status_code=500, detail="审核单写入后读取失败。")

    logger.info(
        "audit=%s suggested=%s narrative_source=%s",
        result.audit_id,
        result.suggested_status.value,
        result.narrative_source.value,
    )
    return result.to_json_dict()


class ExtractRequest(BaseModel):
    filename: str = Field(default="", description="原始文件名，用于后缀判断")
    content_b64: str = Field(default="", description="发票文件的 base64 内容")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "S01_hotel_ok.pdf",
                "content_b64": "JVBERi0xLjQK（PDF 文件的 base64，此处省略）",
            }
        }
    )


@app.post("/api/audit/extract", tags=["核心审核"], summary="票面预抽取（不建单）")
async def audit_extract(req: ExtractRequest):
    """只抽取、不建单 —— 给「票面确认卡」做 AI 预填用。

    返回**全量票面事实 + 每字段置信度**（绿/黄/红/灰，见
    ``finance/extractor.py::assess_confidence``）。回填边界不变：只给
    **票面上有的**；申请人、事由、提交日期这些**申报信息一律不给默认值**。
    """
    if not req.content_b64:
        raise HTTPException(status_code=400, detail="发票内容为空。")
    try:
        blob = base64.b64decode(req.content_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="发票内容不是合法的 base64。")
    if len(blob) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="发票文件过大（上限 8MB）。")
    suffix = Path(req.filename).suffix.lower() or ".pdf"
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix}")

    import tempfile
    from datetime import date

    from finance.extractor import extract
    from finance.policy import load_policy_bundle
    from finance.rules import RuleContext, _resolved_expense_type

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(blob)
            tmp_path = Path(fh.name)
        invoice = extract(tmp_path)
    except Exception as exc:  # noqa: BLE001 —— 抽取失败是可预期的，422 人话不是 500
        raise HTTPException(
            status_code=422, detail=f"未能从票面抽取到字段：{exc}。请对照票面手工填写。"
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)   # 临时文件即用即删，不留孤儿

    # 费用类型**推荐**：复用判定层的关键词归类（rules._resolved_expense_type）——
    # 页面只是预选，申请人可改；判定仍以最终提交的为准。
    policy = load_policy_bundle()
    ctx = RuleContext(
        invoice=invoice,
        request=ReimbursementRequest(
            expense_type="", amount="1", submit_date=date.today()
        ),
        policy=policy,
    )
    resolved, source = _resolved_expense_type(ctx)

    from finance.extractor import CRITICAL_FIELDS, assess_confidence

    return {
        # 全量票面事实 —— 供「票面确认卡」逐项人工核对
        "invoice": {
            "invoice_code": invoice.invoice_code,
            "invoice_number": invoice.invoice_number,
            "invoice_type": invoice.invoice_type,
            "issue_date": invoice.issue_date.isoformat() if invoice.issue_date else None,
            "buyer_name": invoice.buyer_name,
            "buyer_tax_id": invoice.buyer_tax_id,
            "seller_name": invoice.seller_name,
            "seller_tax_id": invoice.seller_tax_id,
            "item_name": invoice.item_name,
            "amount": float(invoice.amount) if invoice.amount is not None else None,
            "tax_rate": invoice.tax_rate,
            "tax_amount": float(invoice.tax_amount) if invoice.tax_amount is not None else None,
            "total": float(invoice.total) if invoice.total is not None else None,
            "total_in_words": invoice.total_in_words,
            "remark": invoice.remark,
        },
        "confidence": assess_confidence(invoice),
        "critical_fields": list(CRITICAL_FIELDS),
        "suggest": {"expense_type": resolved or "", "source": source},
    }


@app.get("/api/audit/{audit_id}", tags=["核心审核"], summary="审核单详情")
async def audit_detail(audit_id: str):
    """单据详情。

    ``advice`` 是**派生字段**：由命中规则的 evidence 现算出来（见
    ``finance/advice.py``），不落库、不参与判定、不影响状态机 —— 它只回答
    「这张单要改成什么样才能过」，把"驳回"变成可执行的动作。
    """
    result = _load_or_404(audit_id)
    return {
        **result.to_json_dict(),
        "advice": suggest_advice(result.findings, policy=load_policy_bundle()),
    }


@app.get("/api/audit/{audit_id}/log", tags=["核心审核"], summary="审计轨迹（哈希链）")
async def audit_log(audit_id: str):
    """审计轨迹。只追加，不可修改 —— 页面上的"可追溯"指的就是这个。

    ``chain`` 字段是哈希链校验结果：任何一行被事后改动都能被发现并定位
    （见 ``finance/store.py::AuditStore.verify_log``）。
    """
    _load_or_404(audit_id)          # 非法/不存在的 id 先挡在门外
    return {
        "events": AUDIT_STORE.read_log(audit_id),
        "chain": AUDIT_STORE.verify_log(audit_id),
    }


@app.post(
    "/api/audit/{audit_id}/decide",
    tags=["核心审核"],
    summary="人工复核决定",
    responses=_DECIDE_RESPONSES,
)
async def audit_decide(audit_id: str, req: AuditDecideRequest):
    """人工决定。**全系统唯一能批准报销的入口。**

    与系统建议相反而未填理由 -> 400（制度 2.3 的服务端强制）。
    """
    result = _load_or_404(audit_id)

    try:
        decision = Decision(req.decision.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="decision 只能是 APPROVED 或 REJECTED。")

    if not req.operator.strip():
        raise HTTPException(status_code=400, detail="必须填写操作人。")

    try:
        decide(
            result,
            decision,
            req.operator.strip(),
            store=AUDIT_STORE,
            override_reason=req.override_reason,
        )
    except OverrideReasonRequired as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except AuditError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # 决定已落盘 → 该单的待办消息一并关闭（由人的决定关闭，不是 agent）
    _agent_runtime().settle(audit_id, by=req.operator.strip())

    return result.to_json_dict()


@app.post("/api/audit/reset", tags=["系统"], summary="清空运行数据（仅演示模式）")
async def audit_reset():
    """清空演示数据 —— **仅演示模式可用**。

    ⚠️ 演示前必须调用一次。排练时跑过的单子会留在查重台账里，
    正式演示再传同一张 S01，R004 会命中"重复报销"，开场全绿基线当场翻车。
    这是本项目最容易踩的一个演示坑，页面上有对应按钮。

    **但默认是关的。** 删除会计凭证与审计轨迹在真实系统里违法（见 ``DEMO_MODE``
    的说明）。要演示就加 ``--demo`` 启动，`start.bat` 已经带上了。
    """
    if not DEMO_MODE:
        raise HTTPException(
            status_code=403,
            detail=(
                "「清空演示数据」已停用：会计凭证与审计轨迹不得删除。"
                "《会计档案管理办法》（财政部、国家档案局令第 79 号）第十四条、"
                "第十五条及附表规定，原始凭证、记账凭证的最低保管期限为 30 年；"
                "本项目的制度 6.2 也写明「留痕记录只追加，不得修改或删除」。"
                "真实系统里正确的更正方式是红冲（生成反向凭证），不是删除。"
                "仅演示时，用 --demo 参数启动可临时开放本接口。"
            ),
        )
    removed = AUDIT_STORE.clear()
    upload_dir = resolve_data_path(UPLOAD_DIR_NAME)
    if upload_dir.is_dir():
        for f in upload_dir.iterdir():
            if f.is_file():
                f.unlink()
    agent_removed = _reset_agent_state()
    logger.info("audit demo reset: %s files removed (agent state: %s)", removed, agent_removed)
    return {"removed": removed, "agent_state_removed": agent_removed}


def _reset_agent_state() -> int:
    """清掉 agent 侧的**运行状态**：事件源账本/收件口、站内收件箱。

    **不碰 ``data/archive/``** —— 它是会计档案，不是运行状态。重置接口的语义
    是"把运行环境恢复到初始"，不是"删除已归档的终稿"；后者正是本项目拒绝做的事。
    """
    removed = 0
    for name in (AGENT_INBOUND_DIR_NAME, "inbox"):
        directory = resolve_data_path(name)
        if not directory.is_dir():
            continue
        for child in directory.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# 受控 agent 层（时间线 / 收件箱 / 恢复 / 事件源）
# ---------------------------------------------------------------------------
#
# 这一节只暴露"看"和"补动作"，**没有任何一个端点能替人签字**：
# 批准仍走 /api/audit/{id}/decide（人 + 理由，推翻要书面说明），
# 归档走 finance/archive.py（人 + 签名，未批不入档）。
# agent 端点最多把单子推到待签批、发消息、补发漏掉的消息。


def _agent_runtime() -> AgentRuntime:
    """每次请求现建一个 runtime —— 它本身无状态，状态全在 store 里。

    现建是为了让收件箱目录**延迟解析**：模块级构造会把 data/ 冻结在 import 时刻
    （store 踩过同一个坑，见 ``AuditStore.base_dir`` 的注释）。
    """
    return AgentRuntime(
        store=AUDIT_STORE,
        inbox=Inbox(base_dir=resolve_data_path("inbox")),
    )


@app.get("/api/agent/actions", tags=["Agent 编排"], summary="动作分级表（L0–L3）")
async def agent_actions():
    """L0–L3 动作表 —— 演示"为什么 agent 不能付款"时直接看这张表。"""
    gate = Gate()
    return {"actions": [gate.explain(action) for action in Action]}


@app.get(
    "/api/agent/{audit_id}/timeline",
    tags=["Agent 编排"],
    summary="审核轨迹时间线",
    responses=_TIMELINE_RESPONSES,
)
async def agent_timeline(audit_id: str):
    """把审计轨迹翻成"谁做了什么"：system / agent / human 三类触发者。"""
    _load_or_404(audit_id)
    return {"audit_id": audit_id, "timeline": _agent_runtime().timeline(audit_id)}


@app.get("/api/agent/inbox", tags=["Agent 编排"], summary="审核台收件箱")
async def agent_inbox(
    who: str = REVIEW_DESK, unread_only: bool = False, open_only: bool = False
):
    """站内收件箱（默认审核台）。不接真邮件/企微 —— 能力先关在系统里。

    ``unread`` 是没打开过的，``open`` 是还没处理的（做完决定就清零）——
    待办看 ``open``，不要拿 ``unread`` 当待办数。
    """
    runtime = _agent_runtime()
    return {
        "who": who,
        "unread": runtime.inbox.count_unread(who),
        "open": runtime.inbox.count_open(who),
        "messages": runtime.inbox.messages_for(
            who, unread_only=unread_only, open_only=open_only
        ),
    }


@app.post("/api/agent/inbox/{message_id}/read", tags=["Agent 编排"], summary="消息标为已读")
async def agent_inbox_read(message_id: str, who: str = REVIEW_DESK):
    if not _agent_runtime().inbox.mark_read(who, message_id):
        raise HTTPException(status_code=404, detail="消息不存在或已读。")
    return {"ok": True}


@app.post("/api/agent/recover", tags=["Agent 编排"], summary="断点恢复（不重跑判定）")
async def agent_recover():
    """断点恢复：只补发漏掉的动作，**不重跑判定**。"""
    return {"recovered": [r.to_dict() for r in _agent_runtime().recover()]}


@app.post("/api/agent/remind", tags=["Agent 编排"], summary="催办超时单据")
async def agent_remind():
    """催办：给超过 48 小时还没处理的单子再发一条提醒。

    **不是定时任务** —— 调用这个端点时才判断哪些单超时；没有后台调度器。
    """
    return {"reminded": _agent_runtime().remind_stale()}


@app.post(
    "/api/agent/scan",
    tags=["Agent 编排"],
    summary="扫描收件口（事件源）",
    responses=_SCAN_RESPONSES,
)
async def agent_scan():
    """扫一轮 ``data/inbound/``（事件源）：成对的文档 + 申请单会被自动受理。

    幂等：同一份材料重复投放只受理一次（事件 id = 内容哈希）。
    """
    source = DirectorySource(
        resolve_data_path(AGENT_INBOUND_DIR_NAME), build_request=_build_request
    )
    try:
        reports = await source.poll_once(_agent_runtime())
    except ScanInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"handled": [r.to_dict() for r in reports], "inbound": str(source.root)}


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------


def _load_or_404(audit_id: str) -> AuditResult:
    """按 id 取审核单；id 非法或不存在一律 404。

    ``store._safe_id`` 会拒绝含非法字符的 id 并抛 ``ValueError`` ——
    路径穿越本身是挡住的，但如果不接这个异常，用户随手敲一个
    ``/api/audit/!!!`` 就会拿到 500 和一条堆栈日志。**用户输入不该打出未捕获异常。**
    """
    try:
        result = AUDIT_STORE.load(audit_id)
    except ValueError:
        result = None
    if result is None:
        raise HTTPException(status_code=404, detail="审核单不存在。")
    return result


def _build_request(spec: dict) -> ReimbursementRequest:
    """把前端传来的 dict 转成报销申请单。缺字段给人话报错，不是 500。

    **必填项在这里拦死，不能只靠页面拦。** 审核台的入口是公开接口，
    谁都能绕过表单直接 POST 一个空对象过来；放过去的话会生成一张
    「申请人空、金额 0」的废单，真的落进查重台账。

    与页面校验的分工：页面拦是为了**立刻**告诉人缺什么，
    这里拦才是**真正**拦得住的那一道。
    """
    from datetime import date

    if not spec:
        raise ValueError("缺少报销申请单数据")

    # 提交日期**不设默认值**。曾经缺省取服务器当天 —— 那等于系统替申请人
    # 编了一个申报日期，而 R003（开票日距提交日 <= 60 天、不得跨年）正是
    # 拿它当判定基准的。申报日期是事实，只能由申请人显式给出。
    submit_raw = spec.get("submit_date")
    if not submit_raw:
        raise ValueError("提交日期未填")
    try:
        submit_date = date.fromisoformat(str(submit_raw))
    except ValueError:
        raise ValueError(f"提交日期格式应为 YYYY-MM-DD，收到 {submit_raw!r}")

    try:
        amount = parse_money(spec.get("amount", 0))
    except ValueError:
        raise ValueError(f"申请金额无法解析：{spec.get('amount')!r}")

    applicant = str(spec.get("applicant") or "").strip()
    department = str(spec.get("department") or "").strip()
    expense_type = str(spec.get("expense_type") or "").strip()

    missing = [
        label
        for label, value in (
            ("申请人", applicant),
            ("所属部门", department),
            ("费用类型", expense_type),
        )
        if not value
    ]
    if missing:
        raise ValueError("以下必填项为空：" + "、".join(missing))

    # 金额 0 不是"小金额"，是**没填**。放过去只会得到一张
    # R010「申请金额 0.00 ≠ 发票 X」的废单 —— 没有财务风险，但污染台账。
    if amount <= 0:
        raise ValueError(f"申请金额必须大于 0，收到 {amount}")

    nights = spec.get("nights")
    headcount = spec.get("headcount")

    return ReimbursementRequest(
        applicant=applicant,
        department=department,
        expense_type=expense_type,
        amount=amount,
        reason=str(spec.get("reason") or "").strip(),
        submit_date=submit_date,
        city=str(spec.get("city") or "").strip(),
        nights=int(nights) if nights not in (None, "") else None,
        headcount=int(headcount) if headcount not in (None, "") else None,
        has_itemized_list=bool(spec.get("has_itemized_list")),
        project=str(spec.get("project") or "").strip(),
        cost_center=str(spec.get("cost_center") or "").strip(),
        allocation_ratio=str(spec.get("allocation_ratio") or "").strip(),
        note=str(spec.get("note") or "").strip(),
    )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="财务报销审核受控 Agent")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址（0.0.0.0 可局域网访问）")
    # 默认 8100 而非 8000：8000 太常用，容易被本机其它本地服务先占上
    parser.add_argument("--port", type=int, default=8100, help="端口")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="演示模式：开放「清空演示数据」。**真实部署不要加** —— "
             "删除会计凭证与审计轨迹违反《会计档案管理办法》的 30 年最低保管要求。",
    )
    args = parser.parse_args()

    global DEMO_MODE
    DEMO_MODE = args.demo

    import uvicorn

    settings = get_settings()
    logger.info("审核台: http://%s:%s/audit", args.host, args.port)
    if DEMO_MODE:
        logger.warning(
            "已开启演示模式 —— 「清空演示数据」可用，它会物理删除审核单与审计轨迹。"
            "真实部署请去掉 --demo。"
        )
    if not settings.api_key:
        logger.warning(
            "未配置 API Key —— 审核功能不受影响，审核意见会退化为模板文字。"
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
