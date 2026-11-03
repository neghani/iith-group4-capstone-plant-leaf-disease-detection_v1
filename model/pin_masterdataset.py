"""Point shared/release.json at the MasterDataset model after its gates pass.

Refuses to edit the pin when test metrics or the PlantDoc overlap report are missing.
Does not invent a threshold or temperature: pass the values chosen on validation.

    python model/pin_masterdataset.py --model-dir model/outputs/masterdataset-resnet18 \\
        --threshold 0.55 --temperature 1.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from common import MASTER_HF_MODEL_ID  # noqa: E402
from publish import masterdataset_publish_gate  # noqa: E402

RELEASE = REPO / "shared" / "release.json"
SPACE_RELEASE = REPO / "hf_space" / "release.json"


def apply_pin(model_dir: Path, threshold: float, temperature: float, revision: str, hub_id: str) -> dict:
    masterdataset_publish_gate(model_dir)
    pin = json.loads(RELEASE.read_text())
    previous = {"hf_model_id": pin["hf_model_id"], "hf_model_revision": pin["hf_model_revision"]}
    pin["hf_model_id"] = hub_id
    pin["hf_model_revision"] = revision
    pin["baseline_revision"] = previous.get("baseline_revision", previous["hf_model_revision"])
    pin["confidence_threshold"] = threshold
    pin["calibration_temperature"] = temperature
    pin["knowledge_file"] = "data/knowledge/diseases.json"
    pin["candidate"] = {
        **{k: previous[k] for k in ("hf_model_id", "hf_model_revision") if k in previous},
        "knowledge_file": "data/knowledge/diseases.plantvillage.json",
        "note": "Previous live pin, kept for rollback.",
    }
    text = json.dumps(pin, indent=2) + "\n"
    RELEASE.write_text(text)
    shutil.copyfile(RELEASE, SPACE_RELEASE)
    print(f"Pinned {hub_id}@{revision}. Previous pin {previous['hf_model_id']}@{previous['hf_model_revision']}.", flush=True)
    return pin


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--revision", default="v1.0")
    p.add_argument("--hub-id", default=MASTER_HF_MODEL_ID)
    args = p.parse_args(argv)
    apply_pin(Path(args.model_dir), args.threshold, args.temperature, args.revision, args.hub_id)


if __name__ == "__main__":
    main()
