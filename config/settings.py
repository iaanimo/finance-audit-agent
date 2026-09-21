"""配置
=======

本项目的全部配置。放在这里的理由：``finance/`` 包用
``from config.settings import get_settings`` 取配置 —— **依赖向下指**，
领域包不自己读环境变量，换宿主环境时只改这一个文件。

配置项很少，都从环境变量读（支持 .env）：

- 数据目录（审核单与上传件落盘位置）
- 叙事模型（把规则结论翻译成人话，**不参与任何判定**）

视觉模型的密钥**不在这里** —— ``tools/vision.py`` 自己读 ``VISION_API_KEY``。
一个密钥只有一处来源，比"两处都写着、只有一处真的生效"安全。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 载入 .env（可选依赖，没装也不影响）
try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    """本项目用到的全部配置。"""

    project_root: Path
    data_dir: Path

    # LLM（仅用于生成审核意见的叙述，不参与任何判定）
    api_key: str
    base_url: str
    model: str

    @property
    def audits_dir(self) -> Path:
        return self.data_dir / "audits"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=PROJECT_ROOT / "data",
        api_key=os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    )
