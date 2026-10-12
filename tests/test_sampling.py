"""50/50 mixed epoch planner: share, inverse-sqrt weights, per-epoch reseed."""

from collections import Counter

from sampling import class_weight_summary, inverse_sqrt_weights, plan_epoch, plan_global


def test_inverse_sqrt_upweights_rare_classes():
    labels = ["a"] * 16 + ["b"] * 4
    w = inverse_sqrt_weights(labels)
    assert abs(sum(w) - 1.0) < 1e-9
    wa, wb = w[0], w[-1]
    # 1/sqrt(16)=0.25, 1/sqrt(4)=0.5, so each rare row is twice a common row.
    assert abs(wb / wa - 2.0) < 1e-9


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
    # Uniform would be 10% rare; 1/sqrt(freq) puts rare near 25% of draws.
    rare_share = picked["rare"] / len(pd_idx)
    assert 0.20 < rare_share < 0.30


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
    # per-row 1/sqrt(freq): a=0.25, b=0.5; class shares 4/6 and 2/6.
    assert abs(by["a"]["inv_sqrt_share"] - 4 / 6) < 1e-9
    assert abs(by["b"]["inv_sqrt_share"] - 2 / 6) < 1e-9
    assert abs(by["b"]["expected_draws"] - 200 * 2 / 6) < 1e-9
