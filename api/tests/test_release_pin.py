"""API Settings field defaults must follow shared/release.json, not leftover v1.0 literals."""

from app.config import Settings
from shared.release import (
    CALIBRATION_TEMPERATURE,
    CONFIDENCE_THRESHOLD,
    CROP_STRATEGY,
    HF_MODEL_ID,
    HF_MODEL_REVISION,
)


def test_settings_defaults_are_the_release_pin():
    fields = Settings.model_fields
    assert fields["hf_model_id"].default == HF_MODEL_ID
    assert fields["hf_model_revision"].default == HF_MODEL_REVISION
    assert fields["confidence_threshold"].default == CONFIDENCE_THRESHOLD
    assert fields["calibration_temperature"].default == CALIBRATION_TEMPERATURE
    assert fields["crop_strategy"].default == CROP_STRATEGY
