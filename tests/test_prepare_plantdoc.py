"""Cropped-PlantDoc preparation on a synthetic checkout: exclusions, cluster split, final test."""

import csv
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_protocol import (
    Item,
    ProtocolError,
    assert_protocol,
    group_by_hash,
    hamming,
    in_validation,
    norm_path,
    phash,
)
from plantdoc_map import MAP, SKIP_CLASS
from prepare_plantdoc import items_from_json, prepare
from smoke import P, build_data

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("prep")
    p = P(root)
    build_data(p, seed=1)
    summary = prepare(p.pd_src, p.pd_train, p.pd_test, p.rw, artifacts_dir=root / "artifacts")
    return p, summary


def _csv(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def _blob(seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (8, 8, 3)).astype(np.uint8)).resize((128, 128), Image.Resampling.BICUBIC)


def test_phash_survives_recompression_and_resize():
    img = _blob(0)
    buf = BytesIO()
    img.resize((100, 100)).save(buf, format="JPEG", quality=60)
    again = Image.open(BytesIO(buf.getvalue()))
    assert hamming(phash(img), phash(again)) <= 6
    assert hamming(phash(img), phash(_blob(1))) > 6


def test_group_by_hash_joins_chains():
    groups = group_by_hash({"a": 0b0, "b": 0b111, "c": 0b111111, "z": (1 << 64) - 1}, max_dist=3)
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["z"] != groups["a"]


def test_validation_share_is_about_12_percent():
    share = sum(in_validation(f"g{i}") for i in range(5000)) / 5000
    assert 0.10 < share < 0.14


def test_norm_path():
    assert norm_path("./train\\Apple Scab Leaf\\x.jpg") == "train/Apple Scab Leaf/x.jpg"


def test_exclusions_logged(prepared):
    p, summary = prepared
    excluded = _csv(p.pd_train / "excluded.csv")
    reasons = {e["reason"] for e in excluded}
    assert {"real_world_set_path", "skip_class"} <= reasons
    assert any("near_dup" in e["reason"] for e in excluded)
    first = sorted(MAP)[0]
    assert any(e["path"] == f"train/{first}/reupload.jpg" for e in excluded)
    assert summary["exclusions"] == {r: sum(e["reason"] == r for e in excluded) for r in reasons}


def test_no_leak_into_train_or_validation(prepared):
    p, _ = prepared
    rows = _csv(p.pd_train / "manifest.csv")
    used = {r["source_path"] for r in rows}
    rw_orig = {r["original_path"] for r in _csv(p.rw / "sources.csv")}
    assert not used & rw_orig
    assert not any(SKIP_CLASS in u for u in used)
    assert not any(u.startswith("test/") for u in used)
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in groups.values())
    assert {"train", "validation"} == {r["split"] for r in rows}


def test_validation_keeps_uncropped_originals(prepared):
    p, _ = prepared
    val = [r for r in _csv(p.pd_train / "manifest.csv") if r["split"] == "validation"]
    assert val and all(r["original_file"] and (p.pd_train / r["original_file"]).exists() for r in val)


def test_final_test_is_plantdoc_test_plus_real_world(prepared):
    p, summary = prepared
    rows = _csv(p.pd_test / "final_test.csv")
    strata = {r["stratum"] for r in rows}
    assert strata == {"plantdoc_test", "real_world_set"}
    n_test = len(MAP) * 2
    assert sum(r["stratum"] == "plantdoc_test" for r in rows) == n_test
    assert sum(r["stratum"] == "real_world_set" for r in rows) == 4
    for r in rows:
        root = p.pd_test if r["source"] == "plantdoc" else p.rw
        assert (root / r["file"]).exists(), r
    assert summary["counts"]["final_test"] == len(rows)


def test_manifest_goes_to_artifacts_dir(prepared):
    p, _ = prepared
    manifest = json.loads((p.root / "artifacts" / "data_manifest.json").read_text())
    assert manifest["plantdoc_commit_pinned"] == "5467f6012d78d1c446145d5f582da6096f852ae8"
    assert manifest["plantdoc_commit"] == ""  # synthetic folder is not a git clone
    assert manifest["counts"]["plantdoc_validation"] > 0


def test_assertions_reload_and_catch_injected_leak(prepared):
    p, _ = prepared
    items = items_from_json(p.pd_train / "items.json")
    assert_protocol(items["train"], items["validation"], items["final_test"])
    leak = items["final_test"][0]
    bad = Item(path="train/x/leak.jpg", split="train", label=leak.label, phash=leak.phash,
               group="g-leak", source="plantdoc")
    with pytest.raises(ProtocolError, match="pHash"):
        assert_protocol(items["train"] + [bad], items["validation"], items["final_test"])
    straddle = Item(path="train/x/s.jpg", split="train", label="x", phash=1, group=items["validation"][0].group,
                    source="plantdoc")
    with pytest.raises(ProtocolError, match="both"):
        assert_protocol(items["train"] + [straddle], items["validation"], items["final_test"])


@pytest.mark.requires_data("model/data/plantdoc_train/items.json")
def test_real_prepared_splits_pass_the_protocol():
    items = items_from_json(REPO / "model/data/plantdoc_train/items.json")
    assert_protocol(items["train"], items["validation"], items["final_test"])
    assert not any(SKIP_CLASS in x.path for x in items["train"] + items["validation"])


@pytest.mark.requires_data("model/data/plantdoc_src/train", "model/data/real_world_set/sources.csv")
def test_real_world_originals_exist_in_the_clone():
    rows = _csv(REPO / "model/data/real_world_set/sources.csv")
    missing = [r["original_path"] for r in rows
               if r.get("original_path") and not (REPO / "model/data/plantdoc_src" / norm_path(r["original_path"])).exists()]
    assert not missing, missing[:5]
