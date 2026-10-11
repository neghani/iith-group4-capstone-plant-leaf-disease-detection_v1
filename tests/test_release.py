"""The live pin is shared/release.json. Do not scatter Hub tags in config defaults."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_release_json_has_the_live_pin():
    from shared.release import (
        BASELINE_REVISION,
        CALIBRATION_TEMPERATURE,
        CONFIDENCE_THRESHOLD,
        CROP_STRATEGY,
        HF_MODEL_ID,
        HF_MODEL_REVISION,
        PIN_PATH,
        as_dict,
    )

    pin = as_dict()
    assert PIN_PATH == REPO / "shared" / "release.json"
    assert HF_MODEL_ID == "neghani/plantvillage-mobilenetv2"
    assert HF_MODEL_REVISION.startswith("v")
    assert BASELINE_REVISION.startswith("v")
    assert HF_MODEL_REVISION != BASELINE_REVISION
    assert 0 < CONFIDENCE_THRESHOLD <= 1
    assert CALIBRATION_TEMPERATURE > 0
    assert CROP_STRATEGY == "B"
    assert pin["hf_model_revision"] == HF_MODEL_REVISION
    space_pin = json.loads((REPO / "hf_space" / "release.json").read_text(encoding="utf-8"))
    assert space_pin == pin, "copy shared/release.json -> hf_space/release.json after bumping the pin"


