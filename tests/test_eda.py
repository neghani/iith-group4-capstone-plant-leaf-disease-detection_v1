"""PlantVillage EDA hard-fails, warnings, and locked default augmentation steps."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from aug import (
    DEFAULT_TRAIN_OPS,
    _norm_aug_policy,
    augmentation_record,
    build_transforms,
    transform_op_names,
)
from eda import _StubProcessor, audit, load_expected_labels, write_class_counts_chart, write_preview

REPO = Path(__file__).resolve().parents[1]
LABELS = load_expected_labels(REPO / "api" / "tests" / "model_labels.json")


def _rgb(path: Path, size: int = 256, color: tuple[int, int, int] = (40, 120, 40), mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (size, size), color if mode == "RGB" else 80).save(path)


def _folder(root: Path, labels: list[str] | None = None, per_class: int = 2, size: int = 256) -> Path:
    labels = labels if labels is not None else LABELS
    for i, name in enumerate(labels):
        for k in range(per_class):
            _rgb(root / name / f"{k}.jpg", size=size, color=((i * 11) % 180 + 30, 90, 50))
    return root


def test_default_transforms_have_eight_steps():
    tf = build_transforms(_StubProcessor(), None, is_training=True)
    assert transform_op_names(tf) == list(DEFAULT_TRAIN_OPS)
    assert len(DEFAULT_TRAIN_OPS) == 8
    assert transform_op_names(tf)[-1] == "Normalize"
    rec = augmentation_record("default")
    assert rec["policy"] == "default"
    assert rec["ops"] == list(DEFAULT_TRAIN_OPS)
    assert rec["pv_rrc_scale"] == [0.7, 1.0]
    assert rec["pd_rrc_scale"] == [0.5, 1.0]


def test_robust_policy_adds_random_erasing_and_is_not_default():
    default = build_transforms(_StubProcessor(), 224, is_training=True, aug_policy="default")
    robust = build_transforms(_StubProcessor(), 224, is_training=True, aug_policy="robust")
    eval_tf = build_transforms(_StubProcessor(), 224, is_training=False, aug_policy="robust")
    assert transform_op_names(default) == list(DEFAULT_TRAIN_OPS)
    assert transform_op_names(robust) == list(DEFAULT_TRAIN_OPS) + ["RandomErasing"]
    assert "RandomErasing" not in transform_op_names(eval_tf)
    assert augmentation_record("robust")["ops"][-1] == "RandomErasing"


def test_unknown_aug_policy_exits():
    with pytest.raises(SystemExit):
        _norm_aug_policy("cutmix")


def test_clean_folder_passes(tmp_path):
    root = _folder(tmp_path / "color")
    payload = audit(root, phash_per_class=0)
    assert payload["ok"]
    assert payload["errors"] == []
    assert payload["n_classes"] == 38
    assert payload["n_images"] == 38 * 2


def test_class_set_mismatch_fails(tmp_path):
    root = _folder(tmp_path / "color", labels=LABELS[:2], per_class=1)
    payload = audit(root, phash_per_class=0)
    assert not payload["ok"]
    assert any("class set" in e for e in payload["errors"])


def test_truncated_file_fails(tmp_path):
    root = _folder(tmp_path / "color", per_class=1)
    (root / LABELS[0] / "0.jpg").write_bytes(b"\xff\xd8\xff")
    payload = audit(root, phash_per_class=0)
    assert not payload["ok"]
    assert any("unreadable" in e for e in payload["errors"])


def test_empty_class_fails(tmp_path):
    root = _folder(tmp_path / "color", per_class=1)
    for path in (root / LABELS[0]).iterdir():
        path.unlink()
    payload = audit(root, phash_per_class=0)
    assert not payload["ok"]
    assert any(f"empty class: {LABELS[0]}" in e for e in payload["errors"])


def test_crop_filter_skips_class_set_check(tmp_path):
    tomatoes = [n for n in LABELS if n.startswith("Tomato___")][:2]
    root = _folder(tmp_path / "color", labels=tomatoes, per_class=1)
    payload = audit(root, crop_filter="Tomato", phash_per_class=0)
    assert payload["ok"]
    assert payload["n_classes"] == 2


def test_warns_on_imbalance_small_non_rgb_and_extra_files(tmp_path):
    root = _folder(tmp_path / "color", per_class=1)
    _rgb(root / LABELS[-1] / "extra_a.jpg")
    _rgb(root / LABELS[-1] / "extra_b.jpg")
    _rgb(root / LABELS[-1] / "extra_c.jpg")
    _rgb(root / LABELS[1] / "tiny.jpg", size=64)
    _rgb(root / LABELS[2] / "gray.png", mode="L")
    (root / LABELS[3] / "notes.txt").write_text("not an image\n")
    payload = audit(root, phash_per_class=0, min_size=224)
    assert payload["ok"]
    codes = {w["code"] for w in payload["warnings"]}
    assert {"imbalance", "small_images", "non_rgb", "extra_files"} <= codes


def test_near_duplicate_sample_is_a_warning(tmp_path):
    root = _folder(tmp_path / "color", per_class=1)
    src = root / LABELS[0] / "0.jpg"
    shutil.copy2(src, root / LABELS[0] / "clone.jpg")
    payload = audit(root, phash_per_class=30)
    assert payload["ok"]
    assert payload["near_duplicates"]["n_pairs"] >= 1
    assert any(w["code"] == "near_duplicates" for w in payload["warnings"])


def test_audit_prints_per_class_progress(tmp_path, capsys):
    tomatoes = [n for n in LABELS if n.startswith("Tomato___")][:2]
    root = _folder(tmp_path / "color", labels=tomatoes, per_class=1)
    audit(root, crop_filter="Tomato", phash_per_class=0)
    out = capsys.readouterr().out
    assert f"[eda] class 1/2 (50%) {tomatoes[0]}: 1 images" in out
    assert "elapsed" in out and "eta" in out
    assert f"[eda] class 2/2 (100%) {tomatoes[1]}: 1 images" in out


def test_class_counts_chart_writes_png(tmp_path):
    payload = {"per_class": [{"name": "Apple___healthy", "n": 3}, {"name": "Tomato___healthy", "n": 1}]}
    path = write_class_counts_chart(payload, tmp_path / "plantvillage_class_counts.png")
    assert path is not None and path.exists()
    img = Image.open(path)
    assert img.size[0] > 0 and img.size[1] > 0


def test_preview_writes_contact_sheet(tmp_path):
    root = _folder(tmp_path / "color", labels=LABELS[:3], per_class=1)
    out = tmp_path / "plantvillage_aug_preview.png"
    by_rel = {name: [f"{name}/0.jpg"] for name in LABELS[:3]}
    path = write_preview(root, LABELS[:3], by_rel, out, n_classes=3, n_views=2, seed=0)
    assert path is not None and path.exists()
    img = Image.open(path)
    assert img.size == (224 * 3, 224 * 3)
