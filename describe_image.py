"""
describe_image.py
=================
借 qwen-vl-max（通义千问视觉模型）"看图"：把本地图片发送到 DashScope 兼容接口，
返回文字描述。给纯文本模型（如 deepseek-v4-flash）补充视觉能力。

用法:
    python describe_image.py <图片路径> [--prompt "你想问关于图的问题"]

API Key 来源（按优先级，与 _load_api_key 的实现一致）:
    1. 环境变量 VISION_API_KEY
    2. $HARNESS_HOME/.credentials.yaml     （把 HARNESS_HOME 指向凭据所在目录）
    3. 与本文件同目录的 .credentials.yaml
    4. 用户主目录下的 .credentials.yaml

    **不硬编码任何本机绝对路径** —— 那既是隐私问题，别人也没法用。
"""

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-vl-max"
DEFAULT_PROMPT = "请详细描述这张图片的内容，包括画面主体、文字、布局和值得注意的细节。"

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _load_api_key() -> str:
    """按顺序找视觉模型密钥：环境变量 -> HARNESS_HOME -> 项目内 -> 用户主目录。"""
    key = os.getenv("VISION_API_KEY", "").strip()
    if key:
        return key
    candidates = []
    if os.getenv("HARNESS_HOME"):
        candidates.append(Path(os.environ["HARNESS_HOME"]) / ".credentials.yaml")
    candidates.append(Path(__file__).resolve().parent / ".credentials.yaml")
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
    raise SystemExit("未找到 VISION_API_KEY（环境变量或 .credentials.yaml）")


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
        raise SystemExit(f"文件不存在: {path}")

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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"解析响应失败: {json.dumps(result, ensure_ascii=False)[:300]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用 qwen-vl-max 描述本地图片")
    parser.add_argument("image", help="图片路径")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="想对图问的问题")
    args = parser.parse_args()
    print(describe(args.image, args.prompt))
