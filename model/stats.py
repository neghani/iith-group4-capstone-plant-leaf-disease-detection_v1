"""Statistics shared by eval_strategies.py, diagnostics.py, calibrate.py and final_test.py.

numpy + scipy only (no torch), so the tests run in the API venv.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np

N_BOOT = 1000
SEED = 0
ECE_BINS = 15
MIN_PRECISION = 0.70
SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 .. 0.95
WIDE_N = 100
WIDE_CI_NOTE = ("With under 100 images the 95% intervals are wide; differences of a few images "
                "are within noise.")


# ---------------------------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def paired_bootstrap(a, b, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Observed mean(a - b) and the 95% percentile interval over resampled images."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def frac(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else f"{k}/0 (n/a)"


def ci(lo: float, hi: float) -> str:
    return f"{100 * lo:.1f}-{100 * hi:.1f}%"


def frac_ci(k: int, n: int) -> str:
    """'12/40 (30.0%), 95% CI 18.1-45.4% (wide)'."""
    if not n:
        return "0/0 (n/a)"
    return f"{frac(k, n)}, 95% CI {ci(*wilson(k, n))}" + (" (wide)" if n < WIDE_N else "")


# ---------------------------------------------------------------------------------------------
# Probabilities and calibration
# ---------------------------------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_probs(probs: np.ndarray) -> np.ndarray:
    """Logits up to a per-row constant: softmax(log p / T) == softmax(z / T)."""
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0))


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (-3.0, 3.0)) -> float:
    """One temperature T minimising NLL, optimised over log T (so T stays positive)."""
    from scipy.optimize import minimize_scalar

    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0:
        raise ValueError("no samples to fit a temperature on")
    res = minimize_scalar(lambda lt: nll(logits, labels, math.exp(lt)), bounds=bounds, method="bounded",
                          options={"xatol": 1e-5})
    return float(math.exp(res.x))


def reliability(conf, correct, bins: int = ECE_BINS) -> list[dict]:
    """Equal-width confidence bins: range, count, mean confidence, accuracy."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        rows.append({"lo": float(lo), "hi": float(hi), "n": n,
                     "confidence": float(conf[m].mean()) if n else None,
                     "accuracy": float(correct[m].mean()) if n else None})
    return rows


def ece(conf, correct, bins: int = ECE_BINS) -> float:
    n = len(conf)
    if n == 0:
        return 0.0
    return float(sum(r["n"] / n * abs(r["accuracy"] - r["confidence"]) for r in reliability(conf, correct, bins) if r["n"]))


def reliability_md(rows: list[dict]) -> list[str]:
    out = ["| Confidence bin | Images | Mean confidence | Accuracy |", "|---|---|---|---|"]
    for r in rows:
        if r["n"]:
            out.append(f"| {r['lo']:.2f}-{r['hi']:.2f} | {r['n']} | {100 * r['confidence']:.1f}% | "
                       f"{100 * r['accuracy']:.1f}% |")
    return out


# ---------------------------------------------------------------------------------------------
# Threshold sweep (pass = reached the model, correct, and confidence >= threshold)
# ---------------------------------------------------------------------------------------------

def sweep_row(conf, correct, reached, threshold: float) -> dict:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    reached = np.asarray(reached, dtype=bool)
    n = len(conf)
    ok = reached & (conf >= threshold)
    passes = int((ok & correct).sum())
    ok_n = int(ok.sum())
    return {"threshold": threshold, "n": n, "passes": passes, "pass_rate": passes / n if n else 0.0,
            "ok_n": ok_n, "coverage": ok_n / n if n else 0.0,
            "precision": passes / ok_n if ok_n else None}


def threshold_sweep(conf, correct, reached, thresholds=SWEEP) -> list[dict]:
    return [sweep_row(conf, correct, reached, t) for t in thresholds]


def select_threshold(table: list[dict], min_precision: float = MIN_PRECISION) -> tuple[dict | None, dict]:
    """(chosen, best_available). chosen is the highest pass rate with precision >= min_precision
    (ties: higher threshold), or None. best_available is the highest-precision row."""
    eligible = [r for r in table if r["ok_n"] > 0 and r["precision"] >= min_precision]
    best = max(table, key=lambda r: (r["precision"] or 0.0, r["pass_rate"], r["threshold"]))
    if not eligible:
        return None, best
    return max(eligible, key=lambda r: (r["pass_rate"], r["threshold"])), best


# ---------------------------------------------------------------------------------------------
# Classification summaries
# ---------------------------------------------------------------------------------------------

def macro_f1_present(y_true: list[str], y_pred: list[str]) -> float:
    """Macro F1 over the classes that occur in y_true (other predicted labels still count as errors)."""
    classes = sorted(set(y_true))
    if not classes:
        return 0.0
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def confusion_pairs(y_true: list[str], y_pred: list[str], k: int = 10) -> list[tuple[str, str, int]]:
    c = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    return [(t, p, n) for (t, p), n in c.most_common(k)]


def absorbing_classes(y_true: list[str], y_pred: list[str], k: int = 10) -> list[tuple[str, int]]:
    """Predicted classes that collect the most wrong answers."""
    return Counter(p for t, p in zip(y_true, y_pred) if t != p).most_common(k)


# ---------------------------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------------------------

def model_key(model: str, revision: str | None) -> str:
    """Identity of the weights under test: the file hash for a local run dir, else id@revision."""
    local = Path(model) / "model.safetensors"
    if local.exists():
        h = hashlib.sha256()
        with open(local, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return "sha256:" + h.hexdigest()
    return f"{model}@{revision}"
