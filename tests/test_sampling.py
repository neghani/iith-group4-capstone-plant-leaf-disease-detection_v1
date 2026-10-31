"""50/50 mixed epoch planner: share, inverse-sqrt weights, per-epoch reseed."""

import math
from collections import Counter

from sampling import class_weight_summary, inverse_sqrt_weights, plan_epoch, plan_global


def test_inverse_sqrt_upweights_rare_classes():
    labels = ["a"] * 16 + ["b"] * 4
    w = inverse_sqrt_weights(labels)
    assert abs(sum(w) - 1.0) < 1e-9
    wa, wb = w[0], w[-1]
    # Class share is 1/sqrt(count): per-row weight is 1/count**1.5.
    # a: 1/16**1.5 = 1/64, b: 1/4**1.5 = 1/8, so each rare row is 8x a common row.
    assert abs(wb / wa - 8.0) < 1e-9


def test_plan_epoch_is_half_plantdoc_and_reproducible():
    labels = ["x"] * 10 + ["y"] * 40
    pairs = plan_epoch(n_pv=100, pd_labels=labels, fraction=0.5, epoch_samples=400, seed=42, epoch=0)
    assert len(pairs) == 400
    n_pd = sum(src == "pd" for src, _ in pairs)
    assert n_pd == 200
    assert all(0 <= i < 100 for src, i in pairs if src == "pv")
    assert all(0 <= i < 50 for src, i in pairs if src == "pd")
    again = plan_epoch(100, labels, 0.5, 400, 42, 0)
    assert pairs == again
    other = plan_epoch(100, labels, 0.5, 400, 42, 1)
    assert pairs != other


def test_rare_plantdoc_class_is_overrepresented():
    labels = ["common"] * 90 + ["rare"] * 10
    pairs = plan_epoch(50, labels, 1.0, 20_000, seed=0, epoch=0)
    pd_idx = [i for src, i in pairs if src == "pd"]
    picked = Counter(labels[i] for i in pd_idx)
    # Uniform would be 10% rare. Class share 1/sqrt(count) puts rare near 75% of draws
    # (1/sqrt(10) vs 1/sqrt(90)). The old per-row 1/sqrt(count) bug put rare near 25%.
    rare_share = picked["rare"] / len(pd_idx)
    assert 0.70 < rare_share < 0.80


def test_plan_global_offsets_into_concatenated_set():
    pv, pd = list(range(5)), list(range(10, 14))
    labels = ["a", "a", "b", "b"]
    idx = plan_global(pv, pd, labels, fraction=0.5, epoch_samples=20, seed=1, epoch=0)
    assert len(idx) == 20
    assert set(idx) <= set(pv) | set(pd)


def test_class_weight_summary_matches_inverse_sqrt():
    labels = ["a"] * 16 + ["b"] * 4
    summary = class_weight_summary(labels, fraction=0.5, epoch_samples=400)
    assert summary["n_pd_draws"] == 200
    assert summary["n_pv_draws"] == 200
    by = {r["class"]: r for r in summary["classes"]}
    assert by["a"]["count"] == 16
    assert by["b"]["count"] == 4
    # Class totals follow 1/sqrt(count): a ∝ 0.25, b ∝ 0.5, shares 1/3 and 2/3.
    assert abs(by["a"]["inv_sqrt_share"] - 1 / 3) < 1e-9
    assert abs(by["b"]["inv_sqrt_share"] - 2 / 3) < 1e-9
    assert abs(by["b"]["expected_draws"] - 200 * 2 / 3) < 1e-6


def test_larger_class_gets_fewer_draws():
    """The PlantDoc chart bug: count 153 drew more than count 74. Class share is 1/sqrt(count)."""
    labels = ["big"] * 153 + ["small"] * 74
    summary = class_weight_summary(labels, fraction=0.5, epoch_samples=4000)
    by = {r["class"]: r for r in summary["classes"]}
    assert by["small"]["expected_draws"] > by["big"]["expected_draws"]
    ratio = by["small"]["inv_sqrt_share"] / by["big"]["inv_sqrt_share"]
    assert abs(ratio - math.sqrt(153 / 74)) < 1e-6
