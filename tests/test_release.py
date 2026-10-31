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
        OOD_THRESHOLD,
        PIN_PATH,
        as_dict,
    )

    pin = as_dict()
    assert PIN_PATH == REPO / "shared" / "release.json"
    assert HF_MODEL_ID == "neghani/plantvillage-mobilenetv2"
    assert HF_MODEL_REVISION == "v1.1"
    assert BASELINE_REVISION.startswith("v")
    assert HF_MODEL_REVISION != BASELINE_REVISION
    assert 0 < CONFIDENCE_THRESHOLD <= 1
    assert CALIBRATION_TEMPERATURE > 0
    assert CROP_STRATEGY == "B"
    assert OOD_THRESHOLD > 0
    assert pin["ood_threshold"] == OOD_THRESHOLD
    assert pin["hf_model_revision"] == HF_MODEL_REVISION
    cand = pin.get("candidate") or {}
    assert cand.get("hf_model_id") == "neghani/plantvillage-resnet18"
    space_pin = json.loads((REPO / "hf_space" / "release.json").read_text(encoding="utf-8"))
    assert space_pin == pin, "copy shared/release.json -> hf_space/release.json after bumping the pin"


def test_env_example_does_not_force_model_overrides():
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    for key in ("HF_MODEL_ID=", "HF_MODEL_REVISION=", "CALIBRATION_TEMPERATURE=", "OOD_THRESHOLD="):
        for line in text.splitlines():
            if line.strip().startswith(key):
                assert line.lstrip().startswith("#"), f"{key} must stay commented in .env.example"


def test_compose_inference_does_not_pass_hf_model_env():
    text = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    # After the inference-service: block, HF_MODEL_* must not be injected.
    block = text.split("inference-service:")[1].split("api-service:")[0]
    assert "HF_MODEL_ID" not in block
    assert "HF_MODEL_REVISION" not in block
