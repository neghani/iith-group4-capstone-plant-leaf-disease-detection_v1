"""Colab GPU DataLoader/AMP flags and local zip staging."""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from common import gpu_train_flags

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from colab_sync import (  # noqa: E402
    DEFAULT_BRANCH,
    DEFAULT_DRIVE_PV_ZIP,
    assert_pv_staging_helper,
    stage_zip_locally,
)


def _flag_value(flags: list[str], name: str) -> str:
    return flags[flags.index(name) + 1]


def test_gpu_train_flags_four_workers_on_high_ram():
    flags = gpu_train_flags(cpu_count=4, cuda_available=True, compute_capability=(7, 5))
    assert _flag_value(flags, "--dataloader_num_workers") == "4"
    assert _flag_value(flags, "--dataloader_pin_memory") == "true"
    assert _flag_value(flags, "--dataloader_persistent_workers") == "true"
    assert "--fp16" in flags
    assert "--bf16" not in flags


def test_gpu_train_flags_eight_workers_on_eight_cores():
    flags = gpu_train_flags(cpu_count=8, cuda_available=True, compute_capability=(7, 5))
    assert _flag_value(flags, "--dataloader_num_workers") == "8"


def test_gpu_train_flags_bf16_on_ampere():
    flags = gpu_train_flags(cpu_count=8, cuda_available=True, compute_capability=(8, 0))
    assert "--bf16" in flags
    assert "--fp16" not in flags


def test_gpu_train_flags_no_amp_on_cpu():
    flags = gpu_train_flags(cpu_count=4, cuda_available=False)
    assert "--fp16" not in flags
    assert "--bf16" not in flags
    assert _flag_value(flags, "--dataloader_num_workers") == "4"


def _write_dataset_zip(path: Path) -> None:
    payload = path.parent / "payload"
    cls = payload / "Tomato___healthy"
    cls.mkdir(parents=True)
    (cls / "a.jpg").write_bytes(b"not-an-image")
    shutil.make_archive(str(path.with_suffix("")), "zip", payload)


def test_stage_zip_locally_copies_before_extract(tmp_path, monkeypatch):
    src = tmp_path / "drive" / "plantvillage.zip"
    src.parent.mkdir()
    _write_dataset_zip(src)
    dest = tmp_path / "plantvillage"
    local_zip = tmp_path / "plantvillage.zip"
    order: list[str] = []
    import colab_sync as cs

    real_copy = cs._copy_with_progress

    def copy2(a, b):
        order.append("copy")
        assert Path(a) == src
        assert Path(b) == local_zip
        return real_copy(a, b)

    orig_zipfile = zipfile.ZipFile

    class TrackingZipFile(orig_zipfile):
        def __init__(self, file, *args, **kwargs):
            order.append("extract")
            assert Path(file) == local_zip
            assert local_zip.exists(), "must copy the zip onto local disk before unzipping"
            super().__init__(file, *args, **kwargs)

    monkeypatch.setattr(cs, "_copy_with_progress", copy2)
    monkeypatch.setattr(zipfile, "ZipFile", TrackingZipFile)

    out = stage_zip_locally(src, dest, local_zip)
    assert out == dest
    assert order == ["copy", "extract"]
    assert local_zip.exists()
    assert (dest / "Tomato___healthy" / "a.jpg").is_file()
    stamp = json.loads((dest / ".zip_stamp.json").read_text())
    assert stamp["size"] == src.stat().st_size


def test_stage_zip_locally_skips_when_stamp_matches(tmp_path):
    src = tmp_path / "drive" / "plantvillage.zip"
    src.parent.mkdir()
    _write_dataset_zip(src)
    dest = tmp_path / "plantvillage"
    local_zip = tmp_path / "plantvillage.zip"
    stage_zip_locally(src, dest, local_zip)
    extracted = dest / "Tomato___healthy" / "a.jpg"
    mtime = extracted.stat().st_mtime_ns
    stage_zip_locally(src, dest, local_zip)
    assert extracted.stat().st_mtime_ns == mtime


def test_stage_zip_locally_rejects_non_zip(tmp_path):
    src = tmp_path / "not_a_zip"
    src.write_text("nope")
    with pytest.raises(SystemExit):
        stage_zip_locally(src, tmp_path / "dest")


def test_default_branch_matches_notebook():
    assert DEFAULT_BRANCH == "main"
    assert DEFAULT_DRIVE_PV_ZIP.endswith("plantvillage.zip")


def test_assert_pv_staging_helper_rejects_stale_module():
    class Stale:
        pass

    with pytest.raises(SystemExit, match="Re-run section 1"):
        assert_pv_staging_helper(Stale())


def test_assert_pv_staging_helper_accepts_current_module():
    import colab_sync

    assert_pv_staging_helper(colab_sync)
