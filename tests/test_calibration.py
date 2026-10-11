"""Temperature scaling, crop mask, NLL fit, threshold selection. No torch."""

import numpy as np
import pytest

from shared.vision.calibration import apply_temperature, apply_temperature_dict, crop_id, crop_mask
from stats import fit_temperature, nll, select_threshold, sweep_row, threshold_sweep


def test_temperature_one_is_identity():
    p = np.array([0.1, 0.2, 0.7])
    out = apply_temperature(p, 1.0)
    np.testing.assert_allclose(out, p, atol=1e-9)


def test_temperature_gt_one_flattens():
    p = np.array([0.05, 0.15, 0.8])
    out = apply_temperature(p, 2.0)
    assert out.max() < p.max()
    assert abs(out.sum() - 1.0) < 1e-9


def test_temperature_dict_matches_array():
    scores = {"a": 0.1, "b": 0.9}
    d = apply_temperature_dict(scores, 1.5)
    arr = apply_temperature(np.array([0.1, 0.9]), 1.5)
    np.testing.assert_allclose([d["a"], d["b"]], arr, atol=1e-9)


def test_crop_mask_keeps_one_crop_and_renormalises():
    names = ["Apple___scab", "Apple___healthy", "Tomato___healthy"]
    p = np.array([0.2, 0.3, 0.5])
    out = crop_mask(p, names, "Apple")
    np.testing.assert_allclose(out, [0.4, 0.6, 0.0], atol=1e-9)
    batch = crop_mask(np.stack([p, p]), names, "Tomato___Early_blight")
    assert batch.shape == (2, 3)
    np.testing.assert_allclose(batch[0], [0.0, 0.0, 1.0], atol=1e-9)


def test_crop_mask_unknown_crop_raises():
    with pytest.raises(ValueError, match="no class matches"):
        crop_mask(np.array([1.0, 0.0]), ["Apple___scab", "Apple___healthy"], "Tomato")


def test_crop_id():
    assert crop_id("Tomato___Early_blight") == "Tomato"
    assert crop_id("Apple") == "Apple"


def test_fit_temperature_does_not_increase_nll():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 5))
    y = rng.integers(0, 5, size=200)
    t_hat = fit_temperature(logits, y)
    assert t_hat > 0
    assert nll(logits, y, t_hat) <= nll(logits, y, 1.0) + 1e-9


def test_select_threshold_70_percent_precision_rule():
    # 10 images: first 4 confident-correct, next 2 confident-wrong, rest below every threshold.
    conf = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.2, 0.1, 0.05, 0.0]
    correct = [True, True, True, True, False, False, True, False, False, False]
    reached = [True] * 10
    table = threshold_sweep(conf, correct, reached)
    chosen, best = select_threshold(table, 0.70)
    assert chosen is not None
    assert chosen["precision"] >= 0.70
    # Higher threshold among tied pass rates is preferred; 0.80 has 4/4 correct.
    assert chosen["threshold"] >= 0.80


def test_select_threshold_provisional_when_nothing_qualifies():
    conf = [0.9, 0.85, 0.8]
    correct = [False, False, True]
    table = threshold_sweep(conf, correct, [True, True, True])
    chosen, best = select_threshold(table, 0.70)
    assert chosen is None
    assert best["precision"] is not None or best["ok_n"] == 0


def test_sweep_row_none_precision_when_no_ok_answers():
    row = sweep_row([0.1, 0.2], [False, False], [True, True], 0.9)
    assert row["ok_n"] == 0
    assert row["precision"] is None
