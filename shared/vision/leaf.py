"""Leaf localisation and scale-normalised sharpness.

Why this exists: the original gate measured Laplacian variance on the whole frame at native
resolution. That number depends on resolution (the same photo scores ~40 at 1024px and ~360
at 256px) and on the background (phone portrait mode blurs it on purpose), so large, clean
photos were rejected as "blurry". Here we find the leaf first, then measure sharpness on the
leaf only, at a fixed normalised size, and use a *re-blur ratio* instead of raw variance:
blur the leaf a little more and see how much high-frequency energy disappears. A sharp leaf
loses most of it (ratio ~0.9+), an already-blurry leaf loses little (ratio <0.75). Unlike raw
variance this does not penalise smooth, glossy leaves (citrus, corn) that are in perfect focus.
The ratio alone is fooled by *heavy* blur (only JPEG noise is left, and noise "disappears" under
re-blur too), so a low absolute-variance floor is kept as a second criterion.

The detector is colour based (vegetation mask + connected components) on purpose: it needs
only numpy/scipy, runs in milliseconds, and keeps the API image free of torch. A CNN detector
can replace `locate_leaves` later without touching the callers.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from shared.vision import config as C


@dataclass
class Sharpness:
    ratio: float  # re-blur ratio 0 (blurry) .. 1 (sharp)
    variance: float  # Laplacian variance at the normalised scale (absolute detail)

    @property
    def rank_score(self) -> float:
        """0.05..1 focus weight used to rank candidates."""
        return float(
            np.clip((self.ratio - 0.5) / 0.4, 0.05, 1.0) * np.clip(self.variance / 60.0, 0.05, 1.0)
        )

    @property
    def passes_gate(self) -> bool:
        return self.ratio >= C.MIN_SHARPNESS_RATIO and self.variance >= C.MIN_LAPLACIAN_VARIANCE


@dataclass
class LeafCandidate:
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in the coordinates of the image searched
    area_frac: float  # leaf pixels / frame pixels
    sharpness: Sharpness
    brightness: float  # mean gray on the leaf
    rank: float = 0.0
    # Leaf mask inside `small_box` (y0, y1, x0, x1) of the MASK_LONG_SIDE copy.
    mask: np.ndarray = field(default=None, repr=False)
    small_box: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class LeafSearch:
    small: Image.Image  # the MASK_LONG_SIDE copy the masks refer to
    scale: tuple[float, float]  # (sx, sy): small -> searched image coordinates
    candidates: list[LeafCandidate]  # best first by `rank`


def laplacian_map(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    return (
        -4 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )


def normalised_sharpness(
    img: Image.Image, mask: np.ndarray | None = None, norm_side: int = C.SHARPNESS_NORM_SIDE
) -> Sharpness:
    """Sharpness at a fixed normalised scale.

    ratio = 1 - var(laplacian(blur(g))) / var(laplacian(g)), after downscaling so the long side
    is at most `norm_side`. Never upscales: a tiny crop cannot fake detail. `mask` (bool, any size)
    restricts the statistic to leaf pixels so a blurred background does not matter.
    """
    work = img
    if max(work.size) > norm_side:
        work = work.copy()
        work.thumbnail((norm_side, norm_side), Image.Resampling.LANCZOS)
    gray = np.asarray(work.convert("L"), dtype=np.float32)
    sel = None
    if mask is not None:
        m = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(work.size, Image.Resampling.NEAREST)
        ) > 0
        m = ndi.binary_erosion(m, iterations=2)  # drop the leaf/background edge itself
        if m.sum() >= 64:
            sel = m
    lap = laplacian_map(gray)
    lap_blur = laplacian_map(ndi.gaussian_filter(gray, C.REBLUR_SIGMA))
    if sel is not None:
        lap, lap_blur = lap[sel], lap_blur[sel]
    base = float(lap.var())
    if base < 1e-6:  # featureless: nothing in focus to speak of
        return Sharpness(0.0, 0.0)
    return Sharpness(float(np.clip(1.0 - lap_blur.var() / base, 0.0, 1.0)), base)


def _vegetation_mask(small: Image.Image) -> np.ndarray:
    hsv = np.asarray(small.convert("HSV"))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= C.VEG_HUE_MIN) & (h <= C.VEG_HUE_MAX) & (s >= C.VEG_SAT_MIN) & (v >= C.VEG_VAL_MIN)
    mask = _drop_out_of_focus(mask, small)
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)), iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)), iterations=2)
    return ndi.binary_fill_holes(mask)


def _drop_out_of_focus(mask: np.ndarray, small: Image.Image) -> np.ndarray:
    """Remove blurred vegetation (portrait-mode bokeh, plants behind the leaf).

    Colour alone cannot tell the held leaf from the grass behind it. Local Laplacian energy
    can: only vegetation within FOCUS_KEEP_RATIO of the sharpest vegetation survives. If the
    whole frame is soft nothing is removed, and the sharpness gate rejects the photo later.
    """
    if not mask.any():
        return mask
    gray = np.asarray(small.convert("L"), dtype=np.float32)
    window = max(5, int(C.FOCUS_WINDOW_FRAC * max(gray.shape)))
    energy = ndi.uniform_filter(np.abs(laplacian_map(gray)), size=window)
    keep = mask & (energy >= C.FOCUS_KEEP_RATIO * np.percentile(energy[mask], 90))
    return keep if keep.sum() >= 0.2 * mask.sum() else mask


def _leaf_regions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label leaf-sized blobs, cutting thin bridges (grass blades, stems, tendrils) between them.

    A plain connected-components pass merged the held leaf with the out-of-focus grass behind
    it. Instead: erode the mask by ~2.5% of the frame (thin things vanish, a leaf survives),
    label the surviving cores, then grow each core back by the same radius inside the mask.
    """
    radius = max(3.0, 0.025 * max(mask.shape))
    core = ndi.distance_transform_edt(mask) > radius
    if not core.any():  # thin-leaved crops (grass-like): keep the plain mask
        return ndi.label(mask)
    core_labels, count = ndi.label(core)
    dist, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    regions = np.where(mask & (dist <= radius), core_labels[iy, ix], 0)
    return regions, count


def locate_leaves(img: Image.Image) -> LeafSearch:
    """Up to MAX_CANDIDATES leaf-like regions, best first. Empty list if nothing green."""
    w, h = img.size
    small = img
    if max(img.size) > C.MASK_LONG_SIDE:
        small = img.copy()
        small.thumbnail((C.MASK_LONG_SIDE, C.MASK_LONG_SIDE), Image.Resampling.BILINEAR)
    sw, sh = small.size
    sx, sy = w / sw, h / sh

    mask = _vegetation_mask(small)
    labels, count = _leaf_regions(mask)
    if count == 0:
        return LeafSearch(small, (sx, sy), [])
    slices = ndi.find_objects(labels)
    frame_px = sw * sh
    gray_small = np.asarray(small.convert("L"), dtype=np.float32)

    candidates: list[LeafCandidate] = []
    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        comp = labels[sl] == idx
        area = int(comp.sum())
        if area / frame_px < C.MIN_COMPONENT_FRAC:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        bbox = (
            int(x0 * sx),
            int(y0 * sy),
            min(w, int(np.ceil(x1 * sx))),
            min(h, int(np.ceil(y1 * sy))),
        )
        cand = LeafCandidate(
            bbox=bbox,
            area_frac=area / frame_px,
            sharpness=normalised_sharpness(img.crop(bbox), comp),
            brightness=float(gray_small[sl][comp].mean()),
            mask=comp,
            small_box=(y0, y1, x0, x1),
        )
        # Prefer a big, sharp leaf near the middle over a blurred plant at the edge.
        cx, cy = (bbox[0] + bbox[2]) / 2 / w, (bbox[1] + bbox[3]) / 2 / h
        centre_dist = min(1.0, float(np.hypot(cx - 0.5, cy - 0.5)) / 0.7071)
        cand.rank = cand.sharpness.rank_score * np.sqrt(cand.area_frac) * (1 - 0.4 * centre_dist)
        candidates.append(cand)

    candidates.sort(key=lambda c: c.rank, reverse=True)
    return LeafSearch(small, (sx, sy), candidates[: C.MAX_CANDIDATES])


def find_leaf_candidates(img: Image.Image) -> list[LeafCandidate]:
    return locate_leaves(img).candidates


def square_window(
    bbox: tuple[int, int, int, int], size: tuple[int, int], fill: float = C.LEAF_FILL
) -> tuple[int, int, int, int]:
    """Square crop window around `bbox` with the leaf filling `fill` of its side.

    The classifier resizes the short side to 256 then centre-crops 224 (87.5%). Measured on 40
    PlantVillage test images (lab and simulated 12 MP field photos): fill 0.8 -> 39/38 correct,
    0.9 -> 40/39, 1.0 -> 37/36 (centre crop clips the leaf). The model was trained on tightly
    framed leaves, so 0.9 wins. The window may extend past the image; callers pad.
    """
    w, h = size
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0, y1 - y0) / fill
    side = max(side, 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = cx - side / 2, cy - side / 2
    # Slide the window back inside the frame when possible, so we pad as little as we can.
    if side <= w:
        left = min(max(left, 0), w - side)
    if side <= h:
        top = min(max(top, 0), h - side)
    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))


def square_crop(img: Image.Image, bbox: tuple[int, int, int, int], fill: float = C.LEAF_FILL) -> Image.Image:
    """`square_window` around bbox, padded with neutral gray, resized once to INFERENCE_SIZE."""
    left, top, right, bottom = square_window(bbox, img.size, fill)
    side = right - left
    canvas = Image.new("RGB", (side, side), C.PAD_COLOR)
    src = img.crop((max(left, 0), max(top, 0), min(right, img.width), min(bottom, img.height)))
    canvas.paste(src, (max(-left, 0), max(-top, 0)))
    return canvas.resize((C.INFERENCE_SIZE, C.INFERENCE_SIZE), Image.Resampling.LANCZOS)
