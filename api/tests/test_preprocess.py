"""Large-photo handling: scale-independent quality gate, best-leaf selection, inference crop.

The scene is synthetic (soil background with portrait-mode blur, one in-focus textured leaf,
one out-of-focus plant behind it) so the tests need no image assets. Thresholds were calibrated
on a real phone photo and 40 PlantVillage images; see docs/challenges-and-approach.md.
"""

from io import BytesIO

import numpy as np
import pytest
from conftest import _upload
from PIL import Image, ImageDraw, ImageFilter

from app.vision import quality
from app.vision.quality import prepare_image
from shared.vision import config as vision_config

BASE_W, BASE_H = 1024, 768
LEAF_BOX = (180, 200, 780, 600)  # x0, y0, x1, y1 of the leaf ellipse in the base scene


def _scene(scale: float = 1.0, leaf_blur: float = 0.0, grass: bool = True) -> Image.Image:
    rng = np.random.default_rng(3)
    soil = np.array([115, 88, 66], dtype=np.float32) + rng.normal(0, 14, (BASE_H, BASE_W, 1))
    bg = Image.fromarray(np.clip(soil, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))

    if grass:  # out-of-focus plant behind the leaf, bottom-right
        blob = Image.new("L", bg.size, 0)
        ImageDraw.Draw(blob).ellipse((800, 560, 1000, 760), fill=255)
        green = Image.new("RGB", bg.size, (70, 130, 60))
        bg.paste(green, mask=blob.filter(ImageFilter.GaussianBlur(14)))

    mask = Image.new("L", bg.size, 0)
    ImageDraw.Draw(mask).ellipse(LEAF_BOX, fill=255)
    tex = rng.normal(0, 22, (BASE_H, BASE_W, 1))
    leaf_arr = np.clip(np.array([55, 135, 45], dtype=np.float32) + tex, 0, 255).astype(np.uint8)
    leaf = Image.fromarray(leaf_arr)
    draw = ImageDraw.Draw(leaf)
    for i in range(8):  # veins
        y = LEAF_BOX[1] + 40 + i * 45
        draw.line((LEAF_BOX[0] + 30, 400, LEAF_BOX[2] - 30, y), fill=(30, 95, 30), width=2)
    img = bg.copy()
    img.paste(leaf, mask=mask)
    if leaf_blur:
        img = img.filter(ImageFilter.GaussianBlur(leaf_blur))
    if scale != 1.0:
        img = img.resize((int(BASE_W * scale), int(BASE_H * scale)), Image.Resampling.LANCZOS)
    return img


def _jpeg(img: Image.Image, quality_=92, **kw) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality_, **kw)
    return buf.getvalue()


def test_large_photo_is_accepted_and_cropped_to_the_leaf():
    big = _jpeg(_scene(scale=4.0))  # 4096 x 3072, like a 12 MP phone shot
    result = prepare_image(big)

    assert result.quality.ok, result.quality.message
    assert result.quality.leaf_found is True
    assert result.quality.original_size == [4096, 3072]
    assert max(result.trace["work_size"]) == vision_config.WORK_MAX_SIDE  # downscaled once, early

    sent = result.candidates[0].crop
    assert sent.size == (vision_config.INFERENCE_SIZE, vision_config.INFERENCE_SIZE)

    # The leaf bbox must sit centred in the square we send, not off in a corner.
    x0, y0, x1, y1 = result.trace["chosen_bbox"]
    work_w = result.trace["work_size"][0]
    s = work_w / BASE_W
    assert x0 == pytest.approx(LEAF_BOX[0] * s, abs=30 * s)
    assert x1 == pytest.approx(LEAF_BOX[2] * s, abs=30 * s)


def test_blurred_grass_is_not_mistaken_for_the_leaf():
    result = prepare_image(_jpeg(_scene()))
    x0, y0, x1, y1 = result.trace["chosen_bbox"]
    assert x1 < 820 and y1 < 640  # grass sits at x>800, y>560; leaf ends at 780 / 600


def test_sharpness_is_resolution_independent():
    small = prepare_image(_jpeg(_scene(scale=1.0))).quality.blur_score
    large = prepare_image(_jpeg(_scene(scale=3.0))).quality.blur_score
    assert small == pytest.approx(large, abs=0.08)


@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_blur_is_rejected_at_any_resolution(scale):
    # Same optical blur, expressed in pixels of each resolution.
    sharp = prepare_image(_jpeg(_scene(scale=scale, leaf_blur=0.0)))
    soft = prepare_image(_jpeg(_scene(scale=scale, leaf_blur=0.0).filter(ImageFilter.GaussianBlur(5 * scale))))
    heavy = prepare_image(_jpeg(_scene(scale=scale, leaf_blur=0.0).filter(ImageFilter.GaussianBlur(12 * scale))))
    assert sharp.quality.ok
    assert not soft.quality.ok and "blurry" in soft.quality.message
    assert not heavy.quality.ok and "blurry" in heavy.quality.message
    assert soft.image is None  # nothing reaches the classifier


def test_old_whole_frame_metric_rejected_big_photos():
    """Regression for the original bug: native-resolution Laplacian variance falls with size."""
    from shared.vision.leaf import laplacian_map

    def old_score(img):
        return float(laplacian_map(np.asarray(img.convert("L"), dtype=np.float32)).var())

    assert old_score(_scene(scale=4.0)) < old_score(_scene(scale=1.0)) / 4
    assert prepare_image(_jpeg(_scene(scale=4.0))).quality.ok


def test_no_green_falls_back_to_whole_frame():
    rng = np.random.default_rng(1)
    gray = Image.fromarray(rng.integers(60, 200, (600, 800, 3), dtype=np.uint8).mean(axis=2).astype(np.uint8)).convert("RGB")
    result = prepare_image(_jpeg(gray))
    assert result.quality.leaf_found is False
    assert result.trace["chosen_bbox"] == [0, 0, 800, 600]


def test_tiny_leaf_in_small_photo_says_move_closer():
    rng = np.random.default_rng(5)
    img = Image.fromarray(np.clip(rng.normal(110, 10, (300, 300, 3)) * [1.0, 0.8, 0.6], 0, 255).astype(np.uint8))
    leaf = np.clip(np.array([55, 135, 45]) + rng.normal(0, 25, (300, 300, 3)), 0, 255).astype(np.uint8)
    mask = Image.new("L", (300, 300), 0)
    ImageDraw.Draw(mask).ellipse((115, 120, 185, 176), fill=255)  # ~70 x 56 px, ~3% of the frame
    img.paste(Image.fromarray(leaf), mask=mask)
    result = prepare_image(_jpeg(img, 95))
    assert result.quality.leaf_found is True
    assert not result.quality.ok and "too small" in result.quality.message


def test_exif_rotated_portrait_photo_is_upright():
    landscape = _scene(scale=1.0)  # 1024 x 768 pixels, stored rotated
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    data = _jpeg(landscape, exif=exif)
    result = prepare_image(data)
    assert result.quality.original_size == [768, 1024]
    assert result.trace["work_size"] == [768, 1024]


def test_pixel_bomb_is_rejected_without_decoding(monkeypatch):
    monkeypatch.setattr(vision_config, "MAX_PIXELS", 1000)
    result = prepare_image(_jpeg(_scene()))
    assert not result.quality.ok and "too many pixels" in result.quality.message


def test_large_file_allowed_up_to_limit():
    assert quality.validate_upload("image/jpeg", 20 * 1024 * 1024, 25 * 1024 * 1024) is None
    err = quality.validate_upload("image/jpeg", 30 * 1024 * 1024, 25 * 1024 * 1024)
    assert err and "25 MB" in err


def test_predict_endpoint_with_large_photo(client, hf_calls, llm_prompts):
    big = _jpeg(_scene(scale=4.0))
    body = client.post("/predict", files=_upload(big)).json()
    assert body["quality"]["ok"] is True
    assert body["report"]["disease_id"] == "Tomato___Early_blight"
    info = body["inference"]
    assert info["model_id"] == "neghani/plantvillage-mobilenetv2"
    assert info["model_revision"] == "v1.0" and info["model_verified"] is True
    assert info["input_size"] == [256, 256]
    assert info["preprocessing"]["original_size"] == [4096, 3072]
    sent = Image.open(BytesIO(hf_calls[0]))
    assert sent.size == (256, 256)
    assert len(hf_calls[0]) < 100_000  # tens of KB over the wire, not megabytes
    assert info["preprocessing"]["crop_strategy"]["strategy"] == "B"


@pytest.mark.parametrize("strategy", ["A", "B", "C", "D", "E"])
def test_crop_strategy_is_one_config_value(monkeypatch, client, hf_calls, llm_prompts, strategy):
    from app.config import get_settings

    monkeypatch.setenv("CROP_STRATEGY", strategy)
    get_settings.cache_clear()
    body = client.post("/predict", files=_upload(_jpeg(_scene()))).json()
    used = body["inference"]["preprocessing"]["crop_strategy"]
    assert used["strategy"] == strategy
    assert len(hf_calls) == used["n_crops"] >= 1  # one classifier call per crop
    assert body["report"]["disease_id"] == "Tomato___Early_blight"


def test_crops_that_disagree_are_flagged_low_confidence(monkeypatch, client, hf_calls, llm_prompts):
    from app.config import get_settings
    from app.vision.hf_client import LabelScore

    monkeypatch.setenv("CROP_STRATEGY", "D")
    get_settings.cache_clear()
    labels = iter(["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold"])
    real_call = hf_calls.__call__

    def varying(image_bytes, content_type="image/jpeg", top_k=None):
        hf_calls.scores = [LabelScore(next(labels, "Tomato___Leaf_Mold"), 0.95)]
        return real_call(image_bytes, content_type, top_k)

    monkeypatch.setattr("app.vision.predictor.classify_image", varying)
    leaves = Image.new("RGB", (900, 600), (205, 205, 200))
    rng = np.random.default_rng(0)
    tex = Image.fromarray(np.clip(np.array([55, 135, 45]) + rng.normal(0, 22, (600, 900, 3)), 0, 255).astype(np.uint8))
    for box in ((30, 80, 280, 330), (330, 300, 580, 560), (630, 60, 870, 300)):
        m = Image.new("L", leaves.size, 0)
        ImageDraw.Draw(m).ellipse(box, fill=255)
        leaves.paste(tex, mask=m)
    body = client.post("/predict", files=_upload(_jpeg(leaves))).json()
    used = body["inference"]["preprocessing"]["crop_strategy"]
    assert used["n_crops"] == 3 and used["agreement"] < 2 / 3
    assert body["low_confidence"] is True
    assert body["report"]["expert_confirmation_required"] is True
