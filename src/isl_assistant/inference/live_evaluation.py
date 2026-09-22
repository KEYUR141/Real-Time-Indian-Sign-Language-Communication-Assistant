"""Structured logging and summaries for prompted live inference trials."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True, slots=True)
class LiveEvaluationRecord:
    elapsed_ms: int
    frame_number: int
    buffer_length: int
    top_label: str | None
    top_probability: float
    second_label: str | None
    second_probability: float
    margin: float
    decision_reason: str
    stable_count: int
    emitted_label: str | None
    hand_coverage: float
    normalization_coverage: float
    model_latency_ms: float
    fps: float


@dataclass(frozen=True, slots=True)
class LiveTrialSummary:
    attempts: int
    expected_label: str | None
    top1_expected_attempts: int
    emitted_labels: tuple[str, ...]
    correct_emissions: int
    false_emissions: int
    first_correct_emission_ms: int | None
    mean_model_latency_ms: float


class LiveTrialTracker:
    """Accumulate inference attempts and calculate a prompted-trial summary."""

    def __init__(self, expected_label: str | None = None) -> None:
        self.expected_label = expected_label
        self._records: list[LiveEvaluationRecord] = []

    def add(self, record: LiveEvaluationRecord) -> None:
        self._records.append(record)

    def summary(self) -> LiveTrialSummary:
        emissions = tuple(
            record.emitted_label
            for record in self._records
            if record.emitted_label is not None
        )
        correct = (
            sum(label == self.expected_label for label in emissions)
            if self.expected_label is not None
            else 0
        )
        false = (
            sum(label != self.expected_label for label in emissions)
            if self.expected_label is not None
            else 0
        )
        first_correct = next(
            (
                record.elapsed_ms
                for record in self._records
                if record.emitted_label == self.expected_label
            ),
            None,
        )
        mean_latency = (
            sum(record.model_latency_ms for record in self._records) / len(self._records)
            if self._records
            else 0.0
        )
        return LiveTrialSummary(
            attempts=len(self._records),
            expected_label=self.expected_label,
            top1_expected_attempts=(
                sum(record.top_label == self.expected_label for record in self._records)
                if self.expected_label is not None
                else 0
            ),
            emitted_labels=emissions,
            correct_emissions=correct,
            false_emissions=false,
            first_correct_emission_ms=first_correct,
            mean_model_latency_ms=mean_latency,
        )


class JsonlEvaluationWriter:
    """Write metadata, prediction attempts, and a summary as JSON Lines."""

    def __init__(self, path: str | Path, metadata: dict[str, Any]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.path.open("w", encoding="utf-8")
        self._write({"event": "metadata", **metadata})

    def _write(self, value: dict[str, Any]) -> None:
        self._handle.write(json.dumps(value, sort_keys=True) + "\n")
        self._handle.flush()

    def write_record(self, record: LiveEvaluationRecord) -> None:
        self._write({"event": "prediction", **asdict(record)})

    def write_summary(self, summary: LiveTrialSummary) -> None:
        value = asdict(summary)
        value["emitted_labels"] = list(summary.emitted_labels)
        self._write({"event": "summary", **value})

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> JsonlEvaluationWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

