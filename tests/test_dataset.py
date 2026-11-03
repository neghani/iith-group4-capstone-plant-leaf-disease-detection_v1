"""MasterDataset loader, PlantDoc overlap, and the publish gate that protects the live pin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset import (
    align_processor,
    assert_class_parity,
    build_dataloaders,
    build_imagefolders,
    build_transforms,
    class_files,
    collect_split_rows,
    resolve_split_root,
)
from download_masterdataset import build_inventory, write_inventory
from knowledge_sync import merge_knowledge, stub_entry
from notebooks.run import require_data_dir
from plantdoc_map import MAP, overlap_map
from plantdoc_ood import build_report, phash_hits
from publish import masterdataset_publish_gate


def _rgb(path: Path, color: tuple[int, int, int] = (40, 140, 50), size: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)


def _tree(root: Path, classes: list[str], val_name: str = "val") -> Path:
    for split in ("train", val_name, "test"):
        for i, name in enumerate(classes):
            _rgb(root / split / name / "a.jpg", color=(20 + i, 100, 40))
            _rgb(root / split / name / "b.jpg", color=(30 + i, 80, 60))
    return root


def test_resolve_split_root_accepts_val_and_a_wrapper(tmp_path):
    classes = ["Mango___healthy", "Tomato___Early_blight"]
    root = _tree(tmp_path / "wrapper" / "inner", classes, val_name="valid")
    found = resolve_split_root(tmp_path)
    assert found["train"] == root / "train"
    assert found["validation"].name == "valid"
    assert found["test"] == root / "test"
    again = resolve_split_root(found["train"])
    assert again["train"] == found["train"]


def test_class_mismatch_and_empty_class_fail(tmp_path):
    classes = ["Mango___healthy", "Tomato___Early_blight"]
    root = _tree(tmp_path / "data", classes)
    (root / "test" / "Mango___healthy").rename(root / "test" / "Mango___Anthracnose")
    with pytest.raises(SystemExit, match="not identical"):
        assert_class_parity({key: class_files(root / name) for key, name in {
            "train": "train", "validation": "val", "test": "test"
        }.items()})
    empty = _tree(tmp_path / "empty", classes)
    for path in (empty / "val" / classes[0]).iterdir():
        path.unlink()
    with pytest.raises(SystemExit, match="empty classes"):
        collect_split_rows(empty)


def test_eval_transform_is_224_and_imagenet_normalized():
    image = Image.new("RGB", (320, 320), (40, 120, 40))
    tensor = build_transforms(False)(image)
    assert tuple(tensor.shape) == (3, 224, 224)
    expected = (40 / 255 - 0.485) / 0.229
    assert tensor[0].mean().item() == pytest.approx(expected, abs=1e-4)
    trained = build_transforms(True)(image)
    assert tuple(trained.shape) == (3, 224, 224)


def test_align_processor_matches_eval_transform():
    """Serving pixels must match build_transforms(False) after align_processor."""
    from types import SimpleNamespace

    from PIL import Image as PILImage

    processor = SimpleNamespace(
        size={"shortest_edge": 256},
        crop_size={"height": 224, "width": 224},
        do_resize=True,
        do_center_crop=True,
        crop_pct=0.875,
        resample=3,
    )
    align_processor(processor)
    assert processor.crop_pct == 1.0
    assert processor.resample == int(PILImage.Resampling.BILINEAR)
    assert processor.size == {"shortest_edge": 224}

    try:
        from transformers import AutoImageProcessor
    except ImportError:
        pytest.skip("transformers not installed")
    real = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
    align_processor(real)
    image = Image.new("RGB", (320, 320), (40, 120, 80))
    tensor = build_transforms(False)(image).numpy()
    served = real(images=image, return_tensors="np")["pixel_values"][0]
    assert float(np.max(np.abs(tensor - served))) < 1e-3


def test_loaders_and_rows_keep_the_given_split(tmp_path):
    classes = ["Mango___healthy", "Tomato___Early_blight"]
    root = _tree(tmp_path / "ds", classes)
    folders = build_imagefolders(root)
    assert folders["train"].classes == folders["validation"].classes == folders["test"].classes == classes
    loaders = build_dataloaders(root, batch_size=2, num_workers=0, pin_memory=False)
    batch = next(iter(loaders["validation"]))
    images, labels = batch
    assert images.shape == (2, 3, 224, 224)
    assert set(int(x) for x in labels) <= {0, 1}
    names, rows, _dirs = collect_split_rows(root, max_samples_per_class=1, seed=1)
    assert names == classes
    assert len(rows["train"]) == 2
    assert all(rel.startswith("train/") for rel, _ in rows["train"])
    assert all(rel.startswith("val/") for rel, _ in rows["validation"])


def test_inventory_records_parity(tmp_path):
    root = _tree(tmp_path / "ds", ["Mango___healthy", "Tomato___Early_blight"])
    payload = build_inventory(root, size_sample=1)
    assert payload["class_sets_identical"]
    assert payload["n_classes"] == 2
    assert payload["n_images"] == 12
    assert payload["size_sample"]["sizes"]["320x320"] == 2
    out = tmp_path / "inventory.json"
    write_inventory(payload, out)
    assert json.loads(out.read_text())["classes"] == payload["classes"]


def test_require_data_dir_accepts_presplit_root(tmp_path, monkeypatch):
    root = _tree(tmp_path / "ds", ["Tomato___healthy"])
    monkeypatch.setenv("DATA_DIR", str(root))
    assert require_data_dir() == root


def test_overlap_keeps_only_shared_classes():
    labels = ["Tomato___Early_blight", "Mango___healthy"]
    mapped = overlap_map(labels)
    assert mapped["Tomato Early blight leaf"] == "Tomato___Early_blight"
    assert "Apple Scab Leaf" not in mapped
    assert overlap_map(list(MAP.values())) == dict(MAP)


def test_overlap_maps_masterdataset_folder_names():
    labels = [
        "apple_apple_scab",
        "healthy_apple",
        "tomato_early_blight",
        "healthy_tomato",
        "corn_gray_leaf_spot",
        "pumpkin_powdery_mildew",
    ]
    mapped = overlap_map(labels)
    assert mapped["Apple Scab Leaf"] == "apple_apple_scab"
    assert mapped["Apple leaf"] == "healthy_apple"
    assert mapped["Tomato Early blight leaf"] == "tomato_early_blight"
    assert mapped["Corn Gray leaf spot"] == "corn_gray_leaf_spot"
    assert "Squash Powdery mildew leaf" not in mapped
    assert "Blueberry leaf" not in mapped


def test_crop_from_masterdataset_names():
    from common import crop_from_class, display_name, is_healthy_class

    assert crop_from_class("Tomato___Early_blight") == "Tomato"
    assert crop_from_class("tomato_early_blight") == "Tomato"
    assert crop_from_class("healthy_bell_pepper") == "Bell Pepper"
    assert crop_from_class("diseased_rice") == "Rice"
    assert display_name("healthy_apple") == "Apple - Healthy"
    assert is_healthy_class("healthy_apple")
    assert is_healthy_class("Mango___healthy")
    assert not is_healthy_class("tomato_early_blight")


def test_phash_flags_a_copied_image_and_ignores_a_different_one(tmp_path):
    shared = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    other = np.zeros((64, 64, 3), dtype=np.uint8)
    other[:, :32] = 255
    master = tmp_path / "master.jpg"
    copy = tmp_path / "copy.jpg"
    different = tmp_path / "other.jpg"
    Image.fromarray(shared).save(master)
    Image.fromarray(shared).save(copy)
    Image.fromarray(other).save(different)
    hits = phash_hits([master], [copy, different])
    assert len(hits) == 1
    assert hits[0]["distance"] == 0


def test_ood_report_counts_images_and_is_not_training_data(tmp_path):
    classes = ["Tomato___Early_blight", "Mango___healthy"]
    master = _tree(tmp_path / "master", classes)
    plantdoc = tmp_path / "plantdoc" / "Tomato Early blight leaf"
    # Use structured noise so pHash does not collide with the solid-colour master tiles.
    noise = np.random.default_rng(7).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    plantdoc.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noise).save(plantdoc / "leaf.jpg")
    report = build_report(
        classes,
        master_root=master,
        plantdoc_root=tmp_path / "plantdoc",
        max_per_class=2,
    )
    assert report["trained_on_plantdoc"] is False
    assert report["n_overlap_classes"] == 1
    assert report["n_images"] == 1
    assert report["per_class"]["Tomato___Early_blight"]["n_images"] == 1
    assert report["ok"] is True
    missing = build_report(classes, plantdoc_root=tmp_path / "absent")
    assert missing["n_images"] == 0
    assert missing["ok"]


def test_knowledge_stubs_do_not_replace_curated_entries():
    existing = {"Tomato___Early_blight": {"display_name": "Tomato Early Blight", "crop": "Tomato"}}
    merged, added = merge_knowledge(existing, ["Tomato___Early_blight", "Mango___healthy"])
    assert added == ["Mango___healthy"]
    assert merged["Tomato___Early_blight"]["display_name"] == "Tomato Early Blight"
    stub = stub_entry("Mango___healthy")
    assert stub["crop"] == "Mango"
    assert stub["curated"] is False
    assert "agronomist" in stub["description"].lower() or "not reviewed" in stub["description"].lower()


def test_masterdataset_gate_requires_counts(tmp_path):
    with pytest.raises(SystemExit, match="publish gate failed"):
        masterdataset_publish_gate(tmp_path)
    (tmp_path / "test_results.json").write_text(json.dumps({"test_accuracy": 0.955}))
    (tmp_path / "run_info.json").write_text(json.dumps({
        "dataset": "masterdataset",
        "split_counts": {"test": 1200},
    }))
    (tmp_path / "plantdoc_ood.json").write_text(json.dumps({
        "n_images": 40,
        "n_overlap_classes": 3,
        "trained_on_plantdoc": False,
        "ok": True,
    }))
    masterdataset_publish_gate(tmp_path)


def test_publish_gate_rejects_plantvillage():
    from publish import publish_gate

    with pytest.raises(SystemExit, match="PlantVillage publishing is retired"):
        publish_gate(Path("/tmp/missing-plantvillage"), "v1.1")


def test_ood_excludes_phash_hits_from_evaluated_set(tmp_path):
    classes = ["Tomato___Early_blight", "Mango___healthy"]
    master = _tree(tmp_path / "master", classes)
    plantdoc_class = tmp_path / "plantdoc" / "Tomato Early blight leaf"
    plantdoc_class.mkdir(parents=True, exist_ok=True)
    # Copy one master image into PlantDoc so pHash flags it; keep a unique second image.
    import shutil

    master_img = next((master / "train" / "Tomato___Early_blight").iterdir())
    shutil.copy(master_img, plantdoc_class / "dup.jpg")
    unique = np.random.default_rng(11).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    Image.fromarray(unique).save(plantdoc_class / "unique.jpg")
    report = build_report(
        classes,
        master_root=master,
        plantdoc_root=tmp_path / "plantdoc",
    )
    assert report["n_excluded"] >= 1
    assert report["ok"] is True
    assert report["n_images"] == report["n_images_before_exclusion"] - report["n_excluded"]
    assert report["phash_hits"] == []
    assert report["phash_hits_before_exclusion"]
