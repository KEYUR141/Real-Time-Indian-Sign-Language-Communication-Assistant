import json
from pathlib import Path

import pytest

from isl_assistant.inference import (
    JsonlEvaluationWriter,
    LiveEvaluationRecord,
    LiveTrialTracker,
)


def record(
    elapsed_ms: int,
    top_label: str | None,
    emitted_label: str | None,
    latency: float,
) -> LiveEvaluationRecord:
    return LiveEvaluationRecord(
        elapsed_ms=elapsed_ms,
        frame_number=elapsed_ms // 10,
        buffer_length=64,
        top_label=top_label,
        top_probability=0.9,
        second_label="old",
        second_probability=0.05,
        margin=0.85,
        decision_reason="candidate",
        stable_count=3,
        emitted_label=emitted_label,
        hand_coverage=1.0,
        normalization_coverage=1.0,
        model_latency_ms=latency,
        fps=25.0,
    )


def test_trial_summary_counts_expected_and_false_emissions() -> None:
    tracker = LiveTrialTracker("hat")
    tracker.add(record(100, "hat", None, 2.0))
    tracker.add(record(200, "hat", "hat", 4.0))
    tracker.add(record(300, "old", "old", 3.0))

    summary = tracker.summary()

    assert summary.attempts == 3
    assert summary.top1_expected_attempts == 2
    assert summary.correct_emissions == 1
    assert summary.false_emissions == 1
    assert summary.first_correct_emission_ms == 200
    assert summary.mean_model_latency_ms == pytest.approx(3.0)


def test_jsonl_writer_records_metadata_prediction_and_summary(tmp_path: Path) -> None:
    path = tmp_path / "trial.jsonl"
    tracker = LiveTrialTracker("hat")
    item = record(100, "hat", "hat", 2.0)
    tracker.add(item)

    with JsonlEvaluationWriter(path, {"trial_id": "hat-01"}) as writer:
        writer.write_record(item)
        writer.write_summary(tracker.summary())

    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [value["event"] for value in values] == ["metadata", "prediction", "summary"]
    assert values[0]["trial_id"] == "hat-01"
    assert values[2]["correct_emissions"] == 1

