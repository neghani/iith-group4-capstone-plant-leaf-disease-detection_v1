"""Progress lines that always reach the notebook cell: elapsed time, percent, ETA.

Used by train.py (a Trainer callback) and eda.py (a loop helper). Prints with flush=True
so Colab shows each line as it happens, even when tqdm bars are silent.
"""

from __future__ import annotations

import os
import time

try:  # eda.py imports this module too and does not need transformers
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover
    TrainerCallback = object


def fmt_secs(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s" if m else f"{s}s"


def eta_secs(done: float, total: float, elapsed: float) -> float | None:
    if done <= 0 or total <= done:
        return 0.0 if total <= done else None
    return elapsed / done * (total - done)


class StepProgress(TrainerCallback):
    """One line every `every` optimizer steps (and at the last step): step, %, epoch, loss, elapsed, ETA."""

    def __init__(self, every: int | None = None) -> None:
        self.every = max(1, int(every or os.environ.get("PROGRESS_EVERY", 10)))
        self.t0 = time.time()

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        print(f"[train] start: {state.max_steps} optimizer steps, log every {self.every}", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        step, total = state.global_step, state.max_steps or 0
        if step % self.every and step != total:
            return
        elapsed = time.time() - self.t0
        eta = eta_secs(step, total, elapsed) if total else None
        loss = next((r["loss"] for r in reversed(state.log_history) if "loss" in r), None)
        pct = f"{100 * step / total:.0f}%" if total else "?"
        loss_s = f"{loss:.4f}" if isinstance(loss, (int, float)) else "-"
        eta_s = fmt_secs(eta) if eta is not None else "-"
        epoch = f"{state.epoch:.2f}" if state.epoch is not None else "-"
        print(f"[train] step {step}/{total} ({pct})  epoch {epoch}  loss {loss_s}  "
              f"elapsed {fmt_secs(elapsed)}  eta {eta_s}", flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            keys = [k for k in metrics if k.endswith("accuracy") or k.endswith("macro_f1")]
            shown = ", ".join(f"{k}={metrics[k]:.4f}" for k in keys if isinstance(metrics[k], (int, float)))
            print(f"[train] eval at step {state.global_step}: {shown or metrics}", flush=True)

