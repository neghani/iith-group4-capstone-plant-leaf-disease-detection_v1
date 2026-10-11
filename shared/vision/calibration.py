"""Temperature scaling and crop-class masking. Probability-space so hosted inference can use them.

    apply_temperature(probs, T)  == softmax(logits / T) when probs = softmax(logits)
    crop_mask(probs, labels, crop)  zeros classes of other crops and renormalises
"""

from __future__ import annotations

import numpy as np


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by T, via softmax(log p / T). T=1 leaves the vector unchanged (renormalised)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    z = np.log(p) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def apply_temperature_dict(scores: dict[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return scores
    labels = list(scores)
    arr = apply_temperature(np.array([scores[k] for k in labels], dtype=np.float64), temperature)
    return {k: float(v) for k, v in zip(labels, arr)}


def crop_id(class_name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato'; already-a-crop names pass through."""
    return class_name.split("___")[0]


def crop_mask(probs: np.ndarray, label_names: list[str], crop: str) -> np.ndarray:
    """Keep only classes of `crop` (the prefix before '___') and renormalise so they sum to 1."""
    key = crop_id(crop)
    keep = np.array([crop_id(n) == key for n in label_names], dtype=bool)
    if not keep.any():
        raise ValueError(f"crop_mask: no class matches crop {crop!r}")
    squeezed = np.asarray(probs, dtype=np.float64)
    single = squeezed.ndim == 1
    if single:
        squeezed = squeezed[None, :]
    out = squeezed.copy()
    out[:, ~keep] = 0.0
    total = out.sum(axis=1, keepdims=True)
    total = np.clip(total, 1e-12, None)
    out = out / total
    return out[0] if single else out
