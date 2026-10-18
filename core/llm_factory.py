"""LLM 工厂
============

**模型在这个项目里只做两件事，都不涉及判定：**

1. 抽取 —— 把非结构化发票变成结构化字段（允许出错，下一层规则会抓）
2. 叙述 —— 把规则引擎的结构化结论翻译成人话（**过护栏，编数字会被丢弃**）

所以这里只支持一种接口（OpenAI 兼容协议），不需要多厂商适配。
DeepSeek / 通义 / 本地 Ollama 都走这个协议，改 base_url 即可。

**注意**：审核结论不经过模型。即使这里完全不可用，审核流程照常跑完，
只是叙述退化成模板文字 —— 这是设计目标，不是降级。
"""

from __future__ import annotations

from typing import Any

from config.settings import get_settings


class LLMNotConfigured(RuntimeError):
    """没有可用的 API Key。调用方应据此降级为模板叙述，而不是让审核失败。"""


def create_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
):
    """创建聊天模型实例。

    :param base_url: 覆盖 ``.env`` 里的地址。用于**指向本地微调模型**
                     （vLLM / llama.cpp / Ollama 的 OpenAI 兼容端口），
                     评测脚本据此把第三路（SFT 后）接到同一套指标上。
    :param api_key: 覆盖 ``.env`` 里的 Key；本地端点通常填 ``dummy`` 即可。
    :raises LLMNotConfigured: 没有配置 API Key（本地端点传 api_key 即可绕过）
    """
    settings = get_settings()

    api_key = api_key or settings.api_key
    if not api_key:
        raise LLMNotConfigured(
            "未配置 API Key。请在 .env 里设置 DEEPSEEK_API_KEY（或 OPENAI_API_KEY）。"
            "审核功能不受影响，叙述会自动退化为模板文字。"
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or settings.model,
        api_key=api_key,
        base_url=base_url or settings.base_url,
        temperature=0.0 if temperature is None else temperature,
        max_tokens=max_tokens or 1024,
        **kwargs,
    )


def get_llm():
    """便捷入口：拿一个默认实例。"""
    return create_llm()
