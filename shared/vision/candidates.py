"""Leaf candidates with the three scores the crop strategies rank by.

    leaf_score     size, focus and centrality of the leaf (leaf.locate_leaves `rank`)
    symptom_score  share of the leaf mask that is not healthy green (see `_symptoms`)
    sharpness      re-blur ratio + Laplacian variance on the leaf (leaf.Sharpness)

If no leaf is found, `find_candidates` returns a single whole-frame candidate with
`leaf_found=False`, so callers never have to special-case an empty list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from shared.vision import config as C
from shared.vision.leaf import LeafCandidate, Sharpness, locate_leaves, normalised_sharpness, square_crop

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in working-image pixels


@dataclass
class Candidate:
    box: Box
    leaf_score: float
    symptom_score: float
    sharpness: Sharpness
    area_frac: float
    brightness: float
    leaf_found: bool = True
    symptom_box: Box | None = None  # largest symptom region, None if there is none
    image: Image.Image = field(default=None, repr=False)

    @cached_property
    def crop(self) -> Image.Image:
        """Classifier-ready square crop (INFERENCE_SIZE) around the leaf."""
        if not self.leaf_found:
            return square_crop(self.image, self.box, fill=1.0)
        return square_crop(self.image, self.box)

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    def trace(self) -> dict:
        return {
            "bbox": list(self.box),
            "area_frac": round(self.area_frac, 3),
            "sharpness": round(self.sharpness.ratio, 3),
            "laplacian_var": round(self.sharpness.variance, 1),
            "rank": round(float(self.leaf_score), 3),
            "symptom_score": round(self.symptom_score, 3),
        }


def symptom_classes(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(healthy_green, symptom) boolean maps for an HSV array (PIL 0..255 channels).

    symptom = yellow | brown | dark | pale/grey, and never healthy green.
    """
    h, s, v = (hsv[..., i].astype(np.int16) for i in range(3))
    healthy = (h >= C.HEALTHY_HUE_MIN) & (h <= C.HEALTHY_HUE_MAX) & (s >= C.HEALTHY_SAT_MIN) & (v >= C.HEALTHY_VAL_MIN)
    dark = v < C.DARK_VAL_MAX
    pale = (s < C.PALE_SAT_MAX) & ~dark
    warm = ((h < C.HEALTHY_HUE_MIN) | (h >= C.WARM_HUE_WRAP)) & ~dark & ~pale
    yellow = warm & (h >= C.YELLOW_HUE_MIN) & (h < C.WARM_HUE_WRAP) & (s >= C.YELLOW_SAT_MIN) & (v >= C.YELLOW_VAL_MIN)
    brown = warm & ~yellow & (s >= C.BROWN_SAT_MIN)
    return healthy, (yellow | brown | dark | pale) & ~healthy


def _symptoms(leaf: LeafCandidate, healthy: np.ndarray, symptom: np.ndarray) -> tuple[float, tuple | None]:
    """(score, small-coords box of the largest region) for one leaf.

    Only symptom regions inside the leaf mask that touch healthy green pixels of the same leaf
    and reach the minimum size count, so soil, pots and background never score.
    """
    y0, y1, x0, x1 = leaf.small_box
    comp = leaf.mask
    leaf_px = int(comp.sum())
    if leaf_px == 0:
        return 0.0, None
    sym = symptom[y0:y1, x0:x1] & comp
    green = healthy[y0:y1, x0:x1] & comp
    labels, n = ndi.label(sym, structure=np.ones((3, 3)))
    if n == 0:
        return 0.0, None
    idx = np.arange(1, n + 1)
    sizes = ndi.sum_labels(np.ones_like(labels), labels, idx)
    touching = np.zeros(n + 1, dtype=bool)
    touching[np.unique(labels[ndi.binary_dilation(green, structure=np.ones((3, 3)))])] = True
    min_px = max(C.SYMPTOM_MIN_REGION_PX, C.SYMPTOM_MIN_REGION_FRAC * leaf_px)
    keep = (sizes >= min_px) & touching[1:]
    if not keep.any():
        return 0.0, None
    score = float(sizes[keep].sum() / leaf_px)
    largest = int(idx[keep][np.argmax(sizes[keep])])
    sl = ndi.find_objects(labels)[largest - 1]
    return score, (y0 + sl[0].start, y0 + sl[0].stop, x0 + sl[1].start, x0 + sl[1].stop)


def find_candidates(img: Image.Image) -> list[Candidate]:
    """Leaf candidates best first by leaf_score; a whole-frame candidate if no leaf is found."""
    search = locate_leaves(img)
    if not search.candidates:
        return [whole_frame_candidate(img)]
    sx, sy = search.scale
    healthy, symptom = symptom_classes(np.asarray(search.small.convert("HSV")))
    out = []
    for leaf in search.candidates:
        score, sbox = _symptoms(leaf, healthy, symptom)
        if sbox is not None:
            y0, y1, x0, x1 = sbox
            sbox = (int(x0 * sx), int(y0 * sy), min(img.width, int(np.ceil(x1 * sx))), min(img.height, int(np.ceil(y1 * sy))))
        out.append(
            Candidate(
                box=leaf.bbox,
                leaf_score=float(leaf.rank),
                symptom_score=score,
                sharpness=leaf.sharpness,
                area_frac=leaf.area_frac,
                brightness=leaf.brightness,
                symptom_box=sbox,
                image=img,
            )
        )
    return out


def whole_frame_candidate(img: Image.Image) -> Candidate:
    return Candidate(
        box=(0, 0, img.width, img.height),
        leaf_score=0.0,
        symptom_score=0.0,
        sharpness=normalised_sharpness(img),
        area_frac=0.0,
        brightness=float(np.asarray(img.convert("L"), dtype=np.float32).mean()),
        leaf_found=False,
        image=img,
    )
