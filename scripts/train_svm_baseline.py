"""Train and validate the manifest-safe INCLUDE SVM baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isl_assistant.config import load_config  # noqa: E402
from isl_assistant.data.baseline_dataset import load_summary_matrix  # noqa: E402
from isl_assistant.data.quality_report import load_expected_features  # noqa: E402
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
        "--output",
        type=Path,
        help="Default: <configured feature output>/models/svm_baseline",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="After validation selection, refit on train+validation and evaluate test once.",
    )
    return parser.parse_args()


def metric_report(y_true: np.ndarray, y_predicted: np.ndarray, labels: list[int]) -> dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_predicted)), 6),
        "macro_f1": round(float(f1_score(y_true, y_predicted, average="macro")), 6),
        "per_class": classification_report(
            y_true,
            y_predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def make_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=c_value, gamma="scale", class_weight="balanced")),
        ]
    )


def write_confusion_matrix(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row.tolist()])


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    subset = args.subset or config.dataset.subset
    output_root = config.output.root.expanduser().resolve()
    experiment_root = (
        args.output.expanduser().resolve()
        if args.output
        else output_root / "models" / "svm_baseline"
    )
    experiment_root.mkdir(parents=True, exist_ok=True)

    expected = load_expected_features(
        config.dataset.manifest_root,
        config.dataset.label_map_path,
        subset,
        excluded_labels=config.dataset.excluded_labels,
    )
    train_x, train_y, train_items = load_summary_matrix(
        output_root, expected, DatasetSplit.TRAIN
    )
    validation_x, validation_y, validation_items = load_summary_matrix(
        output_root, expected, DatasetSplit.VALIDATION
    )
    labels = sorted({item.label_id for item in expected})
    label_names = {
        item.label_id: item.label for item in expected
    }
    candidates: list[dict[str, Any]] = []
    best_model: Pipeline | None = None
    best_c: float | None = None
    best_score = -1.0
    for c_value in (0.1, 1.0, 10.0):
        model = make_pipeline(c_value)
        model.fit(train_x, train_y)
        predicted = model.predict(validation_x)
        metrics = metric_report(validation_y, predicted, labels)
        candidates.append({"C": c_value, **metrics})
        if metrics["macro_f1"] > best_score:
            best_model, best_c, best_score = model, c_value, metrics["macro_f1"]

    assert best_model is not None and best_c is not None
    validation_predicted = best_model.predict(validation_x)
    report: dict[str, Any] = {
        "subset": subset,
        "feature_dimension": int(train_x.shape[1]),
        "train_samples": len(train_items),
        "validation_samples": len(validation_items),
        "test_samples": 0,
        "model": {
            "type": "RBF SVM",
            "class_weight": "balanced",
            "selected_C": best_c,
            "selection_metric": "validation_macro_f1",
        },
        "candidates": candidates,
        "validation": metric_report(validation_y, validation_predicted, labels),
        "test": None,
    }

    model_to_save = best_model
    if args.evaluate_test:
        test_x, test_y, test_items = load_summary_matrix(output_root, expected, DatasetSplit.TEST)
        model_to_save = make_pipeline(best_c)
        model_to_save.fit(
            np.concatenate((train_x, validation_x)),
            np.concatenate((train_y, validation_y)),
        )
        test_predicted = model_to_save.predict(test_x)
        report["test_samples"] = len(test_items)
        report["test"] = metric_report(test_y, test_predicted, labels)
        matrix = confusion_matrix(test_y, test_predicted, labels=labels)
        write_confusion_matrix(
            experiment_root / "test_confusion_matrix.csv",
            matrix,
            [label_names[label] for label in labels],
        )

    joblib.dump(
        {
            "model": model_to_save,
            "label_names": label_names,
            "feature_dimension": int(train_x.shape[1]),
            "subset": subset,
        },
        experiment_root / "svm_baseline.joblib",
    )
    (experiment_root / "svm_baseline_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print("=== SVM BASELINE SUMMARY ===")
    print(f"Selected C: {best_c}")
    print(f"Validation accuracy: {report['validation']['accuracy']:.2%}")
    print(f"Validation Macro F1: {report['validation']['macro_f1']:.2%}")
    if report["test"] is not None:
        print(f"Test accuracy: {report['test']['accuracy']:.2%}")
        print(f"Test Macro F1: {report['test']['macro_f1']:.2%}")
    print(f"Artifacts: {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
