"""Audit INCLUDE feature coverage and create a model-readiness quality report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isl_assistant.config import load_config  # noqa: E402
from isl_assistant.data.quality_report import (  # noqa: E402
    build_quality_report,
    load_expected_features,
)
from isl_assistant.data.schemas import DatasetSplit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "landmarks.toml",
    )
    parser.add_argument("--subset", choices=("include", "include50"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(split.value for split in DatasetSplit),
        default=[split.value for split in DatasetSplit],
    )
    parser.add_argument(
        "--report-name",
        default="dataset_quality_report.json",
        help="Report filename written inside the configured output root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    subset = args.subset or config.dataset.subset
    splits = tuple(DatasetSplit(value) for value in args.splits)
    expected = load_expected_features(
        manifest_root=config.dataset.manifest_root,
        label_map_path=config.dataset.label_map_path,
        subset=subset,
        splits=splits,
        excluded_labels=config.dataset.excluded_labels,
    )
    report = build_quality_report(
        output_root=config.output.root,
        expected=expected,
        sequence_length=config.landmarks.sequence_length,
    )
    report["subset"] = subset
    report["splits"] = [split.value for split in splits]
    report["excluded_labels"] = list(config.dataset.excluded_labels)
    report_path = config.output.root.expanduser().resolve() / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== DATASET QUALITY SUMMARY ===")
    for key in (
        "expected_manifest_samples",
        "valid_manifest_features",
        "missing_manifest_features",
        "invalid_manifest_features",
        "unexpected_features",
    ):
        print(f"{key.replace('_', ' ').title()}: {report[key]}")
    print(f"Expected by split: {report['expected_by_split']}")
    print(f"Valid by split: {report['valid_by_split']}")
    print(f"Report: {report_path}")
    return 1 if report["missing_manifest_features"] or report["invalid_manifest_features"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
