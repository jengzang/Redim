#!/usr/bin/env python3
"""Validate the VoxCeleb layout consumed by the WeSpeaker recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redim_repro.data_validation import ManifestError, validate_voxceleb_layout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="also verify every non-pipeline audio path exists",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    try:
        report = validate_voxceleb_layout(
            args.data_root,
            check_files=args.check_files,
        )
    except ManifestError as exc:
        print(f"data validation failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"data root: {report['root']}")
        print(
            "vox2_dev: "
            f"{report['vox2_dev']['utterances']:,} utterances, "
            f"{report['vox2_dev']['speakers']:,} speakers"
        )
        print(
            "vox1: "
            f"{report['vox1']['utterances']:,} utterances, "
            f"{report['vox1']['speakers']:,} speakers"
        )
        for name, count in sorted(report["vox1"]["trials"].items()):
            print(f"{name}: {count:,} trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
