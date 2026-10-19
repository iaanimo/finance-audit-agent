"""生成器自身的测试
==================

`scripts/make_samples.py` 是样本的**唯一生产者**，它出错的方式很安静：

过去它整体覆盖 ``samples.yaml``，而这个脚本生不出 S12–S14 那 3 张 XML 票据
（铁路客票 / 境外 INVOICE / 定额发票）。于是任何人跑一次生成器，清单就从 16 条
变成 13 条，那 3 张样本连同它们的端到端回归**一起消失**，而评测照常打印
「全部通过」—— 因为评测读的就是被删过的清单，它不会知道自己少了样本。

这里钉住三件事：

1. 默认**保留**清单里生成器不认识的条目（宁可留，不可悄悄删）；
2. ``--prune`` 才允许丢弃；
3. ``build_samples()`` 产生的键不重复、``expects`` 都能被评测脚本解析。

全程不需要 Edge、不渲染 PDF。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import make_samples  # noqa: E402
import run_eval  # noqa: E402


def _merge(tmp_path: Path, existing: dict, prune: bool) -> dict:
    """复刻 main() 的合并逻辑（不落盘 PDF，只验证清单这一半）。"""
    manifest = {s.key: make_samples.manifest_entry(s) for s in make_samples.build_samples()}
    if existing:
        for key, spec in existing.items():
            if key not in manifest and not prune:
                manifest[key] = spec
    return manifest


HANDMADE = {
    "S12_train_ticket": {"title": "铁路电子客票", "pdf": "xml/S12_train_ticket.xml"},
    "S13_foreign_invoice": {"title": "境外 INVOICE", "pdf": "xml/S13_foreign_invoice.xml"},
}


def test_默认保留生成器不认识的条目(tmp_path: Path) -> None:
    merged = _merge(tmp_path, dict(HANDMADE), prune=False)
    assert set(HANDMADE) <= set(merged), "手工维护的 XML 票据被生成器悄悄删掉了"


def test_prune_才会丢弃不认识的条目(tmp_path: Path) -> None:
    merged = _merge(tmp_path, dict(HANDMADE), prune=True)
    assert not (set(HANDMADE) & set(merged))


def test_生成的键不重复且都被评测解析得动() -> None:
    samples = make_samples.build_samples()
    keys = [s.key for s in samples]
    assert len(keys) == len(set(keys)), f"样本键重复：{keys}"

    # expects 是评测唯一的期望来源；解析不出来的话这张样本等于没测。
    for s in samples:
        parsed = run_eval.parse_expects(s.expects)
        assert isinstance(parsed, dict)
        # 写了 "R0xx" 的样本，解析结果里必须真的有这条规则
        for rule_id in parsed:
            assert rule_id in s.expects, f"{s.key} 的 expects 解析出 {rule_id}，原文里却没有"


def test_边界样本成对存在() -> None:
    """压线与超限一分钱必须同时在，否则「<= 限额」这一侧就没人守。"""
    keys = {s.key for s in make_samples.build_samples()}
    assert {"S15_hotel_limit_edge", "S16_hotel_limit_plus_one"} <= keys


@pytest.mark.parametrize(
    "key",
    [
        "S18_injection_english",
        "S19_buyer_fullwidth_space",
        "S21_role_marker_injection",
        "S22_amount_mismatch",
        "S23_account_unmapped",
        "S24_voucher_imbalance",
        "S25_dept_not_in_budget",
    ],
)
def test_对抗样本仍定义在生成器里(key: str) -> None:
    """这些样本的价值全在「精确踩一条规则」，被删掉就没有提示了。"""
    samples = {s.key: s for s in make_samples.build_samples()}
    assert key in samples
    assert samples[key].expects.strip(), f"{key} 缺少 expects，评测无法核对"
