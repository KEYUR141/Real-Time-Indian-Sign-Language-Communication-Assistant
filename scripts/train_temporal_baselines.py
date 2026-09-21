"""Train compact GRU, LSTM, and Transformer landmark-sequence baselines."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isl_assistant.config import load_config  # noqa: E402
from isl_assistant.data.quality_report import load_expected_features  # noqa: E402
from isl_assistant.data.schemas import DatasetSplit  # noqa: E402
from isl_assistant.data.temporal_dataset import load_temporal_arrays  # noqa: E402
from isl_assistant.models.temporal import TemporalClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "landmarks.toml",
    )
    parser.add_argument("--subset", choices=("include", "include50"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("gru", "lstm", "transformer"),
        default=("gru", "lstm"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-feedforward-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Refit the best validation architecture and evaluate the test split once.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def normalize_sequences(
    sequences: np.ndarray,
    lengths: np.ndarray,
    mean: np.ndarray | None = None,
    standard_deviation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize real frames from train-only statistics and retain zero padding."""

    valid = np.arange(sequences.shape[1])[None, :] < lengths[:, None]
    real_frames = sequences[valid]
    if mean is None:
        mean = real_frames.mean(axis=0, dtype=np.float64).astype(np.float32)
    if standard_deviation is None:
        standard_deviation = real_frames.std(axis=0, dtype=np.float64).astype(np.float32)
    safe_standard_deviation = np.where(standard_deviation < 1e-6, 1.0, standard_deviation)
    normalized = np.zeros_like(sequences, dtype=np.float32)
    normalized[valid] = (real_frames - mean) / safe_standard_deviation
    return normalized, mean, safe_standard_deviation


def make_loader(
    sequences: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    dataset = TensorDataset(
        torch.from_numpy(sequences),
        torch.from_numpy(labels),
        torch.from_numpy(lengths),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(
    model: TemporalClassifier,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    actual: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    with torch.no_grad():
        for sequences, labels, lengths in loader:
            logits = model(sequences.to(device), lengths)
            actual.append(labels.numpy())
            predicted.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(actual), np.concatenate(predicted)


def metrics(y_true: np.ndarray, y_predicted: np.ndarray, labels: list[int]) -> dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_predicted)), 6),
        "macro_f1": round(float(f1_score(y_true, y_predicted, average="macro")), 6),
        "per_class": classification_report(
            y_true, y_predicted, labels=labels, output_dict=True, zero_division=0
        ),
    }


def train_one(
    architecture: str,
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    train_lengths: np.ndarray,
    validation_sequences: np.ndarray,
    validation_labels: np.ndarray,
    validation_lengths: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    number_of_classes: int,
) -> tuple[TemporalClassifier, dict[str, Any]]:
    """Train one architecture and restore its best validation-Macro-F1 state."""

    set_seed(args.seed)
    model = TemporalClassifier(
        architecture,
        input_size=train_sequences.shape[2],
        hidden_size=args.hidden_size,
        number_of_classes=number_of_classes,
        dropout=args.dropout,
        maximum_sequence_length=train_sequences.shape[1],
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_feedforward_size=args.transformer_feedforward_size,
    ).to(device)
    train_loader = make_loader(
        train_sequences, train_labels, train_lengths, args.batch_size, shuffle=True
    )
    validation_loader = make_loader(
        validation_sequences,
        validation_labels,
        validation_lengths,
        args.batch_size,
        shuffle=False,
    )
    counts = np.bincount(train_labels, minlength=number_of_classes).astype(np.float32)
    class_weights = counts.sum() / (number_of_classes * np.maximum(counts, 1.0))
    loss_function = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device)
    )
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    labels = list(range(number_of_classes))
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for sequences, targets, lengths in train_loader:
            optimizer.zero_grad()
            logits = model(sequences.to(device), lengths)
            loss = loss_function(logits, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
        actual, predicted = predict(model, validation_loader, device)
        validation_metrics = metrics(actual, predicted, labels)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / len(train_labels), 6),
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )
        if validation_metrics["macro_f1"] > (
            best_metrics["macro_f1"] if best_metrics else -1.0
        ):
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = validation_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    return model, {
        "architecture": architecture,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "validation": best_metrics,
        "history": history,
    }


def fit_fixed_epochs(
    architecture: str,
    sequences: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    number_of_classes: int,
) -> TemporalClassifier:
    """Fit a selected architecture for a preselected epoch count without test access."""

    set_seed(args.seed)
    model = TemporalClassifier(
        architecture,
        input_size=sequences.shape[2],
        hidden_size=args.hidden_size,
        number_of_classes=number_of_classes,
        dropout=args.dropout,
        maximum_sequence_length=sequences.shape[1],
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_feedforward_size=args.transformer_feedforward_size,
    ).to(device)
    loader = make_loader(sequences, labels, lengths, args.batch_size, shuffle=True)
    counts = np.bincount(labels, minlength=number_of_classes).astype(np.float32)
    class_weights = counts.sum() / (number_of_classes * np.maximum(counts, 1.0))
    loss_function = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device)
    )
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    for _ in range(epochs):
        model.train()
        for batch_sequences, targets, batch_lengths in loader:
            optimizer.zero_grad()
            logits = model(batch_sequences.to(device), batch_lengths)
            loss = loss_function(logits, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


def write_confusion_matrix(path: Path, matrix: np.ndarray, names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["actual/predicted", *names])
        for name, row in zip(names, matrix, strict=True):
            writer.writerow([name, *row.tolist()])


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch size must be positive")
    if args.transformer_heads <= 0 or args.transformer_layers <= 0:
        raise ValueError("Transformer heads and layers must be positive")
    if args.transformer_feedforward_size <= 0:
        raise ValueError("Transformer feed-forward size must be positive")
    config = load_config(args.config)
    subset = args.subset or config.dataset.subset
    output_root = config.output.root.expanduser().resolve()
    default_experiment_name = (
        "transformer_baseline"
        if tuple(args.models) == ("transformer",)
        else "temporal_baselines"
    )
    experiment_root = (
        args.output.expanduser().resolve()
        if args.output
        else output_root / "models" / default_experiment_name
    )
    experiment_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    expected = load_expected_features(
        config.dataset.manifest_root,
        config.dataset.label_map_path,
        subset,
        excluded_labels=config.dataset.excluded_labels,
    )
    train_raw, train_y, train_lengths, train_items = load_temporal_arrays(
        output_root, expected, DatasetSplit.TRAIN
    )
    validation_raw, validation_y, validation_lengths, validation_items = load_temporal_arrays(
        output_root, expected, DatasetSplit.VALIDATION
    )
    train_x, mean, standard_deviation = normalize_sequences(train_raw, train_lengths)
    validation_x, _, _ = normalize_sequences(
        validation_raw, validation_lengths, mean, standard_deviation
    )
    label_names = {item.label_id: item.label for item in expected}
    number_of_classes = len(label_names)
    results: list[dict[str, Any]] = []
    trained_models: dict[str, TemporalClassifier] = {}
    for architecture in args.models:
        model, result = train_one(
            architecture,
            train_x,
            train_y,
            train_lengths,
            validation_x,
            validation_y,
            validation_lengths,
            args,
            device,
            number_of_classes,
        )
        trained_models[architecture] = model
        results.append(result)
        torch.save(
            {
                "architecture": architecture,
                "input_size": int(train_x.shape[2]),
                "hidden_size": args.hidden_size,
                "number_of_classes": number_of_classes,
                "dropout": args.dropout,
                "maximum_sequence_length": int(train_x.shape[1]),
                "transformer_heads": args.transformer_heads,
                "transformer_layers": args.transformer_layers,
                "transformer_feedforward_size": args.transformer_feedforward_size,
                "state_dict": model.state_dict(),
                "normalization_mean": mean,
                "normalization_standard_deviation": standard_deviation,
                "label_names": label_names,
            },
            experiment_root / f"{architecture}_validation_best.pt",
        )
        print(
            f"{architecture.upper()} validation Macro F1: "
            f"{result['validation']['macro_f1']:.2%} "
            f"(epoch {result['best_epoch']})"
        )

    best = max(results, key=lambda result: float(result["validation"]["macro_f1"]))
    report: dict[str, Any] = {
        "subset": subset,
        "device": str(device),
        "input_dimension": int(train_x.shape[2]),
        "train_samples": len(train_items),
        "validation_samples": len(validation_items),
        "test_samples": 0,
        "selection_metric": "validation_macro_f1",
        "selected_architecture": best["architecture"],
        "configuration": {
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "transformer_heads": args.transformer_heads,
            "transformer_layers": args.transformer_layers,
            "transformer_feedforward_size": args.transformer_feedforward_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "candidates": results,
        "test": None,
    }
    if args.evaluate_test:
        test_raw, test_y, test_lengths, test_items = load_temporal_arrays(
            output_root, expected, DatasetSplit.TEST
        )
        final_train_raw = np.concatenate((train_raw, validation_raw))
        final_train_y = np.concatenate((train_y, validation_y))
        final_train_lengths = np.concatenate((train_lengths, validation_lengths))
        final_train_x, final_mean, final_standard_deviation = normalize_sequences(
            final_train_raw, final_train_lengths
        )
        test_x, _, _ = normalize_sequences(
            test_raw, test_lengths, final_mean, final_standard_deviation
        )
        final_model = fit_fixed_epochs(
            best["architecture"],
            final_train_x,
            final_train_y,
            final_train_lengths,
            int(best["best_epoch"]),
            args,
            device,
            number_of_classes,
        )
        test_loader = make_loader(
            test_x, test_y, test_lengths, args.batch_size, shuffle=False
        )
        actual, predicted = predict(final_model, test_loader, device)
        labels = list(range(number_of_classes))
        report["test_samples"] = len(test_items)
        report["test"] = metrics(actual, predicted, labels)
        matrix = confusion_matrix(actual, predicted, labels=labels)
        write_confusion_matrix(
            experiment_root / "test_confusion_matrix.csv",
            matrix,
            [label_names[label] for label in labels],
        )
        torch.save(
            {
                "architecture": best["architecture"],
                "input_size": int(train_x.shape[2]),
                "hidden_size": args.hidden_size,
                "number_of_classes": number_of_classes,
                "dropout": args.dropout,
                "maximum_sequence_length": int(train_x.shape[1]),
                "transformer_heads": args.transformer_heads,
                "transformer_layers": args.transformer_layers,
                "transformer_feedforward_size": args.transformer_feedforward_size,
                "state_dict": final_model.state_dict(),
                "normalization_mean": final_mean,
                "normalization_standard_deviation": final_standard_deviation,
                "label_names": label_names,
            },
            experiment_root / "selected_final.pt",
        )

    report_name = (
        "transformer_baseline_report.json"
        if tuple(args.models) == ("transformer",)
        else "temporal_baseline_report.json"
    )
    (experiment_root / report_name).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print("=== TEMPORAL BASELINE SUMMARY ===")
    print(f"Selected architecture: {best['architecture'].upper()}")
    print(f"Validation Macro F1: {best['validation']['macro_f1']:.2%}")
    print(f"Artifacts: {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
