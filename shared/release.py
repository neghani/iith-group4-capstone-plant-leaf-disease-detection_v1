"""Live model pin for the API, Space, and predict scripts.

Edit ``shared/release.json`` when a new Hub tag should become the default.
Environment variables still override (Docker / ``.env``). Historical comparisons
(train-from-v1.0, final-test vs v1.0) use ``baseline_revision``, not this pin.
"""

from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).with_name("release.json")
_PIN = json.loads(_PATH.read_text(encoding="utf-8"))

HF_MODEL_ID: str = _PIN["hf_model_id"]
HF_MODEL_REVISION: str = _PIN["hf_model_revision"]
BASELINE_REVISION: str = _PIN["baseline_revision"]
CONFIDENCE_THRESHOLD: float = float(_PIN["confidence_threshold"])
CALIBRATION_TEMPERATURE: float = float(_PIN["calibration_temperature"])
CROP_STRATEGY: str = _PIN["crop_strategy"]
# Minimum top-1 minus top-2 probability gap. Flat softmax → not_sure even if top-1 is high.
OOD_THRESHOLD: float = float(_PIN.get("ood_threshold", 0.0))
PIN_PATH = _PATH


def as_dict() -> dict:
    return dict(_PIN)
