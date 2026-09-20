"""文件路径约束
================

只有一个函数：把用户给的相对路径解析到 ``data/`` 目录下，**并保证越不出去**。

为什么值得单独留一个模块：上传的发票文件名来自客户端，是不可信输入。
``../../.env`` 或绝对路径这类东西必须被挡在 ``data/`` 之外。
三步拦截：拒绝绝对路径 -> ``resolve()`` 展开 ``..`` -> ``is_relative_to`` 兜底。

（``is_relative_to`` 那一步是防符号链接与 Windows 短路径等"resolve 之后才现形"
的逃逸手法 —— 只做前两步是不够的。）
"""

from __future__ import annotations

from pathlib import Path

# 注意：这里必须**模块级**导入 get_settings，测试才能 monkeypatch 它
# （见 tests/conftest.py 的 temp_project_root fixture）
from config.settings import get_settings


def resolve_data_path(path: str) -> Path:
    """把相对路径解析到 ``<project_root>/data/`` 下。

    :raises ValueError: 传了绝对路径，或路径逃出了 data 目录
    """
    settings = get_settings()
    base_dir = Path(settings.data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    p = Path(path)
    if p.is_absolute():
        raise ValueError(f"不允许绝对路径：{path}")

    resolved = (base_dir / p).resolve()
    base = base_dir.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"路径逃出了 data 目录：{path}")
    return resolved
