"""Compare a candidate model against the MobileNet v1.1 baseline on the final test split.

Baseline macro F1 (284 images, frozen final test) is recorded in model/final_report.md for v1.1.
Run after training:

    python -m model.final_test --model model/outputs/plantvillage-resnet18 --revision_name v1.1 ...

This script prints the baseline and exits non-zero if the candidate final-test macro F1 is lower.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE_REPORT = HERE / "final_report.md"
MOBILENET_V11_MACRO_F1 = 0.545  # from final_report.md v1.1 table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate_macro_f1", type=float, required=True, help="Macro F1 from final_test for the ResNet run.")
    p.add_argument("--baseline_macro_f1", type=float, default=MOBILENET_V11_MACRO_F1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"MobileNet v1.1 baseline macro F1: {args.baseline_macro_f1:.1%}")
    print(f"ResNet-18 candidate macro F1:       {args.candidate_macro_f1:.1%}")
    if args.candidate_macro_f1 <= args.baseline_macro_f1:
        print("Gate FAILED: candidate must beat MobileNet v1.1 macro F1 on the frozen final split.")
        sys.exit(1)
    print("Gate PASSED: candidate beats MobileNet v1.1 on macro F1.")
    out = HERE / "outputs" / "gate_resnet18.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "baseline_model": "neghani/plantvillage-mobilenetv2@v1.1",
                "baseline_macro_f1": args.baseline_macro_f1,
                "candidate_macro_f1": args.candidate_macro_f1,
                "passed": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
