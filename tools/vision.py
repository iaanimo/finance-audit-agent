"""视觉模型客户端（qwen-vl-max / DashScope 兼容接口）
=====================================================

把本地图片发给通义千问视觉模型，拿回文字描述。本项目里只有一个用途：
``finance/extractor.py`` 的兜底抽取 —— PDF 没有文本层时，把页面里的位图
交给它读成字段。

用法::

    ./.venv/Scripts/python.exe -m tools.vision <图片路径> [--prompt "想问的问题"]

API Key 来源（按优先级，与 :func:`_load_api_key` 的实现一致）::

    1. 环境变量 VISION_API_KEY
    2. $HARNESS_HOME/.credentials.yaml   （把 HARNESS_HOME 指向凭据所在目录）
    3. 项目根目录下的 .credentials.yaml
    4. 用户主目录下的 .credentials.yaml

**不硬编码任何本机绝对路径** —— 那既是隐私问题，别人也没法用。

报错一律用 :class:`VisionError`。**这里刻意不用 ``SystemExit``**：
``SystemExit`` 继承 ``BaseException``，``except Exception`` 接不住，
会顺着调用栈把 uvicorn 进程一起带走 —— 一个"没配密钥"的配置问题，
不该表现为"服务没了"。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-vl-max"
DEFAULT_PROMPT = "请详细描述这张图片的内容，包括画面主体、文字、布局和值得注意的细节。"


class VisionError(RuntimeError):
    """视觉模型不可用（缺密钥、文件不存在、网络或响应异常）。"""


#: 项目根目录 —— 凭据查找用得到（见 :func:`_load_api_key`）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _load_api_key() -> str:
    """按顺序找视觉模型密钥：环境变量 -> HARNESS_HOME -> 项目根 -> 用户主目录。"""
    key = os.getenv("VISION_API_KEY", "").strip()
    if key:
        return key
    candidates = []
    if os.getenv("HARNESS_HOME"):
        candidates.append(Path(os.environ["HARNESS_HOME"]) / ".credentials.yaml")
    # 项目根目录：tools/ 的上一层。这个文件原来在仓库根目录，搬家后路径要跟着走，
    # 否则"项目内的 .credentials.yaml"这一档会悄悄失效（找不到密钥却不报错的那种失效）。
    candidates.append(PROJECT_ROOT / ".credentials.yaml")
    candidates.append(Path.home() / ".credentials.yaml")
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            name, _, value = line.partition(":")
            if name.strip() == "VISION_API_KEY":
                return value.strip().strip('"').strip("'")
    raise VisionError(
        "未找到 VISION_API_KEY（环境变量或 .credentials.yaml）。"
        "不配也能跑：PDF 文本层路径不依赖视觉模型，只是扫描件无法兜底。"
    )


def mime_of(path: Path) -> str:
    return MIME_BY_EXT.get(path.suffix.lower(), "image/png")


def describe(
    image_path: str,
    prompt: str = DEFAULT_PROMPT,
    timeout: int = 60,
    max_tokens: int = 1024,
) -> str:
    api_key = _load_api_key()
    path = Path(image_path)
    if not path.exists():
        raise VisionError(f"文件不存在: {path}")

    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    data_uri = f"data:{mime_of(path)};base64,{b64}"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except VisionError:
        raise
    except Exception as exc:  # noqa: BLE001 —— 网络/HTTP/解析异常统一成 VisionError
        raise VisionError(f"调用视觉模型失败：{type(exc).__name__}: {exc}") from exc

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"解析响应失败: {json.dumps(result, ensure_ascii=False)[:300]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用 qwen-vl-max 描述本地图片")
    parser.add_argument("image", help="图片路径")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="想对图问的问题")
    args = parser.parse_args(argv)
    try:
        print(describe(args.image, args.prompt))
    except VisionError as exc:
        # 命令行工具给人看的一句话，而不是一个堆栈
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
