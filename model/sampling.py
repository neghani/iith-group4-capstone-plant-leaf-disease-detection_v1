"""Torch-free mixed-batch planner: 50% PlantDoc / 50% PlantVillage, PlantDoc 1/sqrt(freq).

train.py wraps plan_epoch in a thin torch Sampler. Tests call this module directly.
"""

from __future__ import annotations

import math
import random
from collections import Counter


def inverse_sqrt_weights(labels: list) -> list[float]:
    """Per-row sampling weight: 1/sqrt(class frequency), normalised to sum to 1."""
    counts = Counter(labels)
    raw = [1.0 / math.sqrt(counts[lab]) for lab in labels]
    total = sum(raw)
    if total <= 0:
        n = len(labels) or 1
        return [1.0 / n] * len(labels)
    return [w / total for w in raw]


def plan_epoch(
    n_pv: int,
    pd_labels: list,
    fraction: float,
    epoch_samples: int,
    seed: int,
    epoch: int,
) -> list[tuple[str, int]]:
    """One epoch of (`'pv'|'pd'`, local index) draws. Re-seeded per epoch so resume is deterministic.

    `fraction` is the PlantDoc share of the epoch. Draws are with replacement.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"plantdoc_fraction must be in [0, 1], got {fraction}")
    if epoch_samples <= 0:
        raise ValueError("epoch_samples must be positive")
    if n_pv <= 0:
        raise ValueError("PlantVillage train is empty")
    if not pd_labels:
        raise ValueError("PlantDoc train is empty")
    rng = random.Random(int(seed) + 10007 * int(epoch))
    n_pd = int(round(epoch_samples * fraction))
    n_pd = min(max(n_pd, 0), epoch_samples)
    n_pv_draw = epoch_samples - n_pd
    pv = [rng.randrange(n_pv) for _ in range(n_pv_draw)]
    weights = inverse_sqrt_weights(pd_labels)
    pd = rng.choices(range(len(pd_labels)), weights=weights, k=n_pd)
    pairs = [("pv", i) for i in pv] + [("pd", i) for i in pd]
    rng.shuffle(pairs)
    return pairs


def plan_global(
    pv_indices: list[int],
    pd_indices: list[int],
    pd_labels: list,
    fraction: float,
    epoch_samples: int,
    seed: int,
    epoch: int,
) -> list[int]:
    """Map source-local draws onto indices of a concatenated train set."""
    local = plan_epoch(len(pv_indices), pd_labels, fraction, epoch_samples, seed, epoch)
    out = []
    for src, i in local:
        out.append(pv_indices[i] if src == "pv" else pd_indices[i])
    return out
