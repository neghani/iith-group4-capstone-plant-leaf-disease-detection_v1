"""Shared preprocessing: leaf candidates, symptom score and crop strategies A-E.

Synthetic scenes (textured leaves on a plain background) and a fake predictor, so no model or
image assets are needed. Run from the repo root: `python -m pytest tests`.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from shared.vision.candidates import find_candidates
from shared.vision.strategies import STRATEGY_IDS, decide_status, run_strategy

W, H = 800, 600
GREEN = (55, 135, 45)
BROWN = (120, 72, 35)
EXPECTED_KEYS = {"status", "disease", "confidence", "top_k", "n_crops", "agreement", "leaf_found", "strategy"}


def _scene(leaves, background=(205, 205, 200), soil=False, seed=0) -> Image.Image:
    """`leaves`: list of (ellipse box, spots) where spots is a list of (cx, cy, r) brown lesions."""
    rng = np.random.default_rng(seed)
    if soil:
        arr = np.array([115, 88, 66], dtype=np.float32) + rng.normal(0, 12, (H, W, 1))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    else:
        img = Image.new("RGB", (W, H), background)
    tex = np.clip(np.array(GREEN, dtype=np.float32) + rng.normal(0, 22, (H, W, 1)), 0, 255).astype(np.uint8)
    leaf_img = Image.fromarray(tex)
    spot_tex = np.clip(np.array(BROWN, dtype=np.float32) + rng.normal(0, 12, (H, W, 1)), 0, 255).astype(np.uint8)
    spot_img = Image.fromarray(spot_tex)
    for box, spots in leaves:
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).ellipse(box, fill=255)
        img.paste(leaf_img, mask=mask)
        smask = Image.new("L", (W, H), 0)
        for cx, cy, r in spots:
            ImageDraw.Draw(smask).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        img.paste(spot_img, mask=smask)
    return img


ONE_LEAF = [((200, 150, 600, 450), [])]
THREE_LEAVES = [((40, 60, 270, 300), []), ((300, 280, 530, 540), []), ((560, 50, 780, 290), [])]


class FakePredictor:
    """Returns `dists[i]` for the i-th image of each call (last one repeated)."""

    def __init__(self, dists):
        self.dists = dists
        self.calls: list[int] = []

    def predict_proba(self, images):
        self.calls.append(len(images))
        return [dict(self.dists[min(i, len(self.dists) - 1)]) for i in range(len(images))]


AGREE = FakePredictor([{"Tomato___Early_blight": 0.9, "Tomato___healthy": 0.1}])


def test_single_green_leaf_on_plain_background_gives_one_candidate():
    cands = find_candidates(_scene(ONE_LEAF))
    assert len(cands) == 1
    c = cands[0]
    assert c.leaf_found is True
    x0, y0, x1, y1 = c.box
    assert abs(x0 - 200) < 25 and abs(x1 - 600) < 25 and abs(y0 - 150) < 25 and abs(y1 - 450) < 25
    assert c.crop.size == (256, 256)
    assert c.sharpness.passes_gate
    assert c.symptom_score < 0.01
    for attr in ("box", "crop", "leaf_score", "symptom_score", "sharpness"):
        assert hasattr(c, attr)


def test_no_green_falls_back_to_whole_frame():
    rng = np.random.default_rng(1)
    gray = rng.integers(60, 200, (H, W), dtype=np.uint8)
    cands = find_candidates(Image.fromarray(gray).convert("RGB"))
    assert len(cands) == 1
    assert cands[0].leaf_found is False
    assert cands[0].box == (0, 0, W, H)


def test_symptom_score_counts_lesions_not_background():
    spotted = [((200, 150, 600, 450), [(330, 250, 28), (460, 330, 32), (400, 220, 22)])]
    healthy_on_soil = find_candidates(_scene(ONE_LEAF, soil=True))[0]
    diseased = find_candidates(_scene(spotted))[0]
    assert healthy_on_soil.leaf_found and healthy_on_soil.symptom_score < 0.01  # soil is outside the leaf
    assert diseased.symptom_score > 0.03
    assert diseased.symptom_box is not None


def test_symptom_strategies_pick_the_blemished_leaf():
    # Big healthy leaf in the centre (B's choice) and a smaller spotted leaf at the side.
    leaves = [((150, 120, 560, 480), []), ((610, 200, 780, 420), [(680, 280, 16), (720, 350, 14)])]
    img = _scene(leaves)
    b = run_strategy(img, "B", AGREE)
    c = run_strategy(img, "C", AGREE)
    e = run_strategy(img, "E", AGREE)
    assert b["crop_boxes"][0][0] < 300
    assert c["crop_boxes"][0][0] > 600
    ex0, ey0, ex1, ey1 = e["crop_boxes"][0]
    assert 610 <= (ex0 + ex1) / 2 <= 780 and (ex1 - ex0) < 120  # tight patch on the spotted leaf


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
def test_each_strategy_returns_expected_keys(strategy):
    res = run_strategy(_scene(THREE_LEAVES), strategy, AGREE, threshold=0.5)
    assert EXPECTED_KEYS <= res.keys()
    assert res["status"] == "ok" and res["disease"] == "Tomato___Early_blight"
    assert res["top_k"][0] == {"disease": "Tomato___Early_blight", "confidence": 0.9}
    assert res["n_crops"] == (3 if strategy == "D" else 1)
    assert res["agreement"] == 1.0


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
def test_gate_rejection_has_the_same_keys(strategy):
    blurry = _scene(ONE_LEAF).filter(ImageFilter.GaussianBlur(6))
    fake = FakePredictor([{"x": 1.0}])
    res = run_strategy(blurry, strategy, fake)
    assert EXPECTED_KEYS <= res.keys()
    assert res["status"] == "retake_photo" and res["gate_reason"] == "blurry" and res["n_crops"] == 0
    assert fake.calls == []  # nothing reaches the classifier


def test_strategy_d_is_not_sure_when_crops_disagree():
    # Averaged top-1 is X at 0.62 (above threshold), but only 1 of 3 crops has X as top-1.
    fake = FakePredictor([
        {"X": 0.95, "Y": 0.05},
        {"Y": 0.55, "X": 0.45},
        {"Z": 0.55, "X": 0.45},
    ])
    res = run_strategy(_scene(THREE_LEAVES), "D", fake, threshold=0.5)
    assert res["n_crops"] == 3
    assert res["disease"] == "X" and res["confidence"] > 0.5
    assert res["agreement"] == pytest.approx(1 / 3)
    assert res["status"] == "not_sure"


def test_strategy_d_ok_when_two_of_three_agree():
    fake = FakePredictor([{"X": 0.9, "Y": 0.1}, {"X": 0.8, "Y": 0.2}, {"Y": 0.6, "X": 0.4}])
    res = run_strategy(_scene(THREE_LEAVES), "D", fake, threshold=0.5)
    assert res["agreement"] == pytest.approx(2 / 3)
    assert res["status"] == "ok"


def test_flip_tta_classifies_each_crop_twice():
    fake = FakePredictor([{"X": 0.9, "Y": 0.1}])
    res = run_strategy(_scene(THREE_LEAVES), "D", fake, tta=True)
    assert fake.calls == [6] and res["n_crops"] == 3


def test_decide_status():
    assert decide_status(0.9, 1.0, 0.5) == "ok"
    assert decide_status(0.4, 1.0, 0.5) == "not_sure"
    assert decide_status(0.9, 0.5, 0.5) == "not_sure"
    assert decide_status(0.9, None, 0.5) == "ok"
