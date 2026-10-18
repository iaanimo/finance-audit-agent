"""SFT 数据生成的测试
====================

微调数据最容易出的两种错，都是静默的：

1. **训练/评测不同分布** —— 造数据时自己另写一份提示词，模型学到的是"另一种问法"，
   评测分再高也不作数。这里钉住两条路径必须逐字一致。
2. **标注口径漂移** —— 标签不是规则引擎给的，而是另写一段"我觉得应该是驳回"。
   这里钉住标签来自 ``finance.rules.evaluate`` + ``aggregate``。

另外钉住两件本次实现时踩到的事：

- 没有历史台账视图时，R004/R012 会降级成 WARN，**于是没有任何一张样本能判通过**，
  训练集里会缺掉 APPROVED 这一类（第一次跑数据时就是这样）；
- 评测集（27 张）的发票号码绝不能出现在训练数据里。
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_sft_data as b  # noqa: E402
from finance.judge_prompt import build_judge_prompt  # noqa: E402
from finance.policy import load_policy_bundle  # noqa: E402

SAMPLE_DIR = PROJECT_ROOT / "finance" / "samples"


def test_两条路径的提示词逐字一致() -> None:
    """SFT 造数据用的提示词 == 评测脚本用的提示词（同一函数，这里再钉一遍）。"""
    rng = random.Random(7)
    invoice, request = b.make_case("ok", rng, 1)
    prompt = build_judge_prompt(invoice, request)
    assert "公司报销制度要点" in prompt
    assert invoice.invoice_number in prompt
    assert str(request.amount) in prompt
    assert prompt.rstrip().endswith("}"), "提示词必须以下发输出格式的 JSON 模板结尾"


def test_标签来自规则引擎而不是另写逻辑() -> None:
    policy = load_policy_bundle()
    rng = random.Random(11)

    ok_inv, ok_req = b.make_case("ok", rng, 2)
    assert b.label(ok_inv, ok_req, policy)["verdict"] == "通过"

    bad_inv, bad_req = b.make_case("wrong_buyer", rng, 3)
    assert b.label(bad_inv, bad_req, policy)["verdict"] == "驳回"


def test_大写金额带零位() -> None:
    assert b._words(1040) == "壹仟零肆拾圆整"
    assert b._words(520) == "伍佰贰拾圆整"


def test_同一随机种子的输出可复现() -> None:
    a = b.make_case("hotel_over", random.Random(2026), 9)
    c = b.make_case("hotel_over", random.Random(2026), 9)
    assert a[0].model_dump() == c[0].model_dump()


def test_评测集的发票号码不得进入训练数据() -> None:
    manifest = yaml.safe_load((SAMPLE_DIR / "samples.yaml").read_text(encoding="utf-8"))
    held = {(spec.get("invoice") or {}).get("invoice_number", "") for spec in manifest.values()}
    rng = random.Random(5)
    for i in range(1, 120):
        invoice, _ = b.make_case("ok", rng, i)
        assert invoice.invoice_number not in held


def test_端到端能生成且三类结论都有(tmp_path: Path) -> None:
    """跑一次真实生成（每种 1 条），确认三个结论类别都存在、文件格式正确。"""
    out = tmp_path / "sft"
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_sft_data.py"),
         "--per-type", "1", "--out-dir", str(out), "--seed", "3"],
        check=True, capture_output=True,
    )
    train = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    val = [json.loads(line) for line in (out / "val.jsonl").read_text(encoding="utf-8").splitlines()]
    assert train and val

    verdicts = {r["meta"]["verdict"] for r in train + val}
    assert verdicts == {"通过", "驳回", "转人工"}, f"结论类别不全：{verdicts}"

    first = (train + val)[0]
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant"]
    target = json.loads(first["messages"][2]["content"])
    assert set(target) == {"verdict", "reason", "hit"}
    assert (out / "stats.md").is_file()
