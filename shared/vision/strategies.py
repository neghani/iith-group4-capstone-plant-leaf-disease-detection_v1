"""Crop strategies behind one interface, so the API switches with a single config value.

    A  whole frame, padded square and resized (no cropping)
    B  crop around the best leaf by leaf_score (size, focus, centrality); the original behaviour
    C  crop around the leaf with the highest symptom_score
    D  top 3 leaves by (SYMPTOM_RANK_FLOOR + symptom_score) * sharpness; softmax averaged,
       "not_sure" when fewer than MIN_AGREEMENT of the crops agree with the averaged top-1
    E  tight patch around the largest symptom region only (control; expected to do worse)

C, D and E only consider leaves that pass the gate's sharpness test (all leaves if none does),
so a symptom-heavy but out-of-focus leaf behind the subject is never picked.

`run_strategy` runs the quality gate first; every strategy sees the same gate decision.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol

from PIL import Image, ImageOps

from shared.vision import config as C
from shared.vision.candidates import Box, Candidate, find_candidates
from shared.vision.leaf import square_crop
from shared.vision.quality import GateResult, check_gate

STRATEGY_IDS = ("A", "B", "C", "D", "E")
DESCRIPTIONS = {
    "A": "Whole frame, resized",
    "B": "Sharpest leaf candidate (current)",
    "C": "Most symptomatic leaf candidate",
    "D": "Top-3 leaves, averaged softmax + agreement",
    "E": "Tight patch on strongest symptom (control)",
}


class Predictor(Protocol):
    def predict_proba(self, images: Sequence[Image.Image]) -> list[dict[str, float]]:
        """One {label: probability} dict per image (the full softmax, or at least its top-k)."""


Crops = list[tuple[Image.Image, Box]]


def _sharp(cands: list[Candidate]) -> list[Candidate]:
    leaves = [c for c in cands if c.leaf_found and c.sharpness.passes_gate]
    return leaves or cands


def _most_symptomatic(cands: list[Candidate]) -> Candidate:
    return max(_sharp(cands), key=lambda c: (c.symptom_score, c.leaf_score))


def whole_frame(img: Image.Image, cands: list[Candidate]) -> Crops:
    box = (0, 0, img.width, img.height)
    return [(square_crop(img, box, fill=1.0), box)]


def sharpest_leaf(img: Image.Image, cands: list[Candidate]) -> Crops:
    return [(cands[0].crop, cands[0].box)]


def most_symptomatic_leaf(img: Image.Image, cands: list[Candidate]) -> Crops:
    c = _most_symptomatic(cands)
    return [(c.crop, c.box)]


def top3_ensemble(img: Image.Image, cands: list[Candidate]) -> Crops:
    ranked = sorted(
        _sharp(cands),
        key=lambda c: ((C.SYMPTOM_RANK_FLOOR + c.symptom_score) * c.sharpness.rank_score, c.leaf_score),
        reverse=True,
    )
    return [(c.crop, c.box) for c in ranked[:3]]


def symptom_patch(img: Image.Image, cands: list[Candidate]) -> Crops:
    c = _most_symptomatic(cands)
    if c.symptom_box is None:  # nothing symptomatic: same as C
        return [(c.crop, c.box)]
    x0, y0, x1, y1 = c.symptom_box
    side = max(max(x1 - x0, y1 - y0) / C.SYMPTOM_PATCH_FILL, C.SYMPTOM_PATCH_MIN_FRAC * max(c.width, c.height), 8)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    return [(square_crop(img, box, fill=1.0), box)]


SELECTORS: dict[str, Callable[[Image.Image, list[Candidate]], Crops]] = {
    "A": whole_frame,
    "B": sharpest_leaf,
    "C": most_symptomatic_leaf,
    "D": top3_ensemble,
    "E": symptom_patch,
}


def decide_status(
    confidence: float,
    agreement: float | None,
    threshold: float,
    *,
    margin: float | None = None,
    ood_threshold: float = 0.0,
) -> str:
    if agreement is not None and agreement < C.MIN_AGREEMENT - 1e-9:
        return "not_sure"
    if ood_threshold > 0 and margin is not None and margin < ood_threshold - 1e-9:
        return "not_sure"
    return "ok" if confidence >= threshold else "not_sure"


def _mean(dists: list[dict[str, float]]) -> dict[str, float]:
    labels = {k for d in dists for k in d}
    return {k: sum(d.get(k, 0.0) for d in dists) / len(dists) for k in labels}


def _top1(dist: dict[str, float]) -> str:
    return max(dist.items(), key=lambda kv: (kv[1], kv[0]))[0]


def retake_result(gate: GateResult, strategy_id: str, threshold: float = C.DEFAULT_THRESHOLD) -> dict:
    return {
        "status": "retake_photo",
        "disease": None,
        "confidence": None,
        "top_k": [],
        "n_crops": 0,
        "agreement": None,
        "strategy": strategy_id,
        "threshold": threshold,
        "leaf_found": gate.leaf_found,
        "gate_reason": gate.reason,
        "message": gate.message,
        "crop_boxes": [],
    }


def run_strategy(
    img: Image.Image,
    strategy_id: str,
    predictor: Predictor,
    *,
    threshold: float = C.DEFAULT_THRESHOLD,
    ood_threshold: float = 0.0,
    top_k: int = 3,
    original_size: tuple[int, int] | list[int] | None = None,
    candidates: list[Candidate] | None = None,
    gate: GateResult | None = None,
    tta: bool = False,
) -> dict:
    """Gate -> strategy crops -> classifier -> decision. Same keys as diagnose, plus n_crops/agreement.

    `img` is the decoded working copy (quality.load_working_copy). `candidates` and `gate` may
    be passed in when the caller already computed them for the same image.
    `tta` also classifies each crop mirrored left-right and averages the two.
    """
    if strategy_id not in SELECTORS:
        raise ValueError(f"Unknown strategy {strategy_id!r}; expected one of {STRATEGY_IDS}")
    started = time.perf_counter()
    if candidates is None:
        candidates = find_candidates(img)
    if gate is None:
        gate = check_gate(img, candidates, original_size)
    if not gate.ok:
        return retake_result(gate, strategy_id, threshold)

    picked = SELECTORS[strategy_id](img, candidates)
    crops = [c for c, _ in picked]
    n = len(crops)
    probs = predictor.predict_proba(crops + [ImageOps.mirror(c) for c in crops] if tta else crops)
    per_crop = [_mean([probs[i], probs[i + n]]) for i in range(n)] if tta else list(probs)
    avg = _mean(per_crop)
    ranked = sorted(avg.items(), key=lambda kv: (-kv[1], kv[0]))
    disease, confidence = ranked[0]
    second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    margin = float(confidence) - second
    agreement = sum(_top1(p) == disease for p in per_crop) / n
    return {
        "status": decide_status(
            confidence, agreement, threshold, margin=margin, ood_threshold=ood_threshold
        ),
        "disease": disease,
        "confidence": float(confidence),
        "margin": float(margin),
        "top_k": [{"disease": k, "confidence": round(float(v), 4)} for k, v in ranked[:top_k]],
        "n_crops": n,
        "agreement": agreement,
        "strategy": strategy_id,
        "threshold": threshold,
        "ood_threshold": ood_threshold,
        "leaf_found": gate.leaf_found,
        "gate_reason": None,
        "message": None,
        "crop_boxes": [list(b) for _, b in picked],
        "crop_tops": [_top1(p) for p in per_crop],
        "tta": tta,
        "elapsed_s": round(time.perf_counter() - started, 4),
    }
