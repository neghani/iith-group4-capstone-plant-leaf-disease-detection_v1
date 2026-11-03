"""Add knowledge stubs for MasterDataset classes that are not curated yet.

Existing entries in diseases.json are left unchanged. New ids get a placeholder that
says an agronomist has not reviewed it. This does not run until the inventory exists,
and it does not replace the live 38-class file unless ``--write`` is passed.

    python model/knowledge_sync.py --inventory data/inventory.json --out model/outputs/knowledge_pending.json
    python model/knowledge_sync.py --inventory data/inventory.json --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from common import crop_from_class, display_name, is_healthy_class  # noqa: E402

DEFAULT_KNOWLEDGE = REPO / "api" / "data" / "knowledge" / "diseases.json"
DEFAULT_LABELS = REPO / "api" / "tests" / "model_labels.json"


def stub_entry(class_name: str) -> dict:
    crop = crop_from_class(class_name)
    healthy = is_healthy_class(class_name)
    condition = display_name(class_name)
    return {
        "display_name": condition,
        "crop": crop,
        "category": "healthy" if healthy else "unknown",
        "description": (
            f"Placeholder for {condition}. This class comes from the MasterDataset folder name. "
            "An agronomist has not reviewed it."
        ),
        "symptoms": [
            {
                "name": "See the photo",
                "explanation": "Symptoms for this class are not curated yet.",
            }
        ],
        "alternatives": [],
        "actions": [
            {
                "step": f"Confirm {condition} with a local agriculture officer before any treatment.",
                "safety_note": "Do not apply chemicals from this placeholder.",
            }
        ],
        "monitoring": "Recheck the plant and take a new photo if it changes.",
        "limitations": "Auto-generated from the class name. Not curated agronomy advice.",
        "curated": False,
    }


def merge_knowledge(existing: dict, class_names: list[str]) -> tuple[dict, list[str]]:
    merged = dict(existing)
    added = []
    for name in class_names:
        if name in merged:
            continue
        merged[name] = stub_entry(name)
        added.append(name)
    return merged, added


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", default=str(REPO / "data" / "inventory.json"))
    p.add_argument("--knowledge", default=str(DEFAULT_KNOWLEDGE))
    p.add_argument("--labels", default=str(DEFAULT_LABELS), help="Updated only with --write.")
    p.add_argument("--out", default=str(HERE / "outputs" / "knowledge_pending.json"))
    p.add_argument("--write", action="store_true", help="Replace diseases.json and model_labels.json.")
    args = p.parse_args(argv)
    inventory_path = Path(args.inventory)
    if not inventory_path.is_file():
        raise SystemExit(
            f"{inventory_path} is missing. Download the MasterDataset and write the inventory first. "
            "The live knowledge base was not changed."
        )
    inventory = json.loads(inventory_path.read_text())
    classes = list(inventory["classes"])
    knowledge_path = Path(args.knowledge)
    existing = json.loads(knowledge_path.read_text()) if knowledge_path.is_file() else {}
    merged, added = merge_knowledge(existing, classes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"Stubs added: {len(added)}. Kept curated entries: {len(existing)}. Wrote {out}", flush=True)
    if not args.write:
        print("Live diseases.json was not changed. Re-run with --write after the new model passes its gates.", flush=True)
        return
    live = {name: merged[name] for name in classes}
    knowledge_path.write_text(json.dumps(live, indent=2) + "\n")
    Path(args.labels).write_text(json.dumps(classes, indent=2) + "\n")
    print(f"Updated {knowledge_path} and {args.labels}", flush=True)


if __name__ == "__main__":
    main()
