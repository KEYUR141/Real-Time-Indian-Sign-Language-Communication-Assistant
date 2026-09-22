"""Unknown rejection and temporal acceptance for live predictions."""

from __future__ import annotations

from dataclasses import dataclass

from .rolling_buffer import BufferSnapshot
from .temporal_predictor import Prediction


@dataclass(frozen=True, slots=True)
class RejectionConfig:
    minimum_frames: int = 48
    minimum_hand_coverage: float = 0.5
    minimum_normalization_coverage: float = 0.8
    minimum_probability: float = 0.75
    minimum_margin: float = 0.2

    def __post_init__(self) -> None:
        if self.minimum_frames <= 0:
            raise ValueError("minimum_frames must be positive")
        for name in (
            "minimum_hand_coverage",
            "minimum_normalization_coverage",
            "minimum_probability",
            "minimum_margin",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    """One prediction after readiness and unknown-rejection checks."""

    label: str | None
    probability: float
    margin: float
    reason: str

    @property
    def accepted(self) -> bool:
        return self.label is not None


class UnknownRejector:
    def __init__(self, config: RejectionConfig | None = None) -> None:
        self.config = config or RejectionConfig()

    def evaluate(
        self,
        snapshot: BufferSnapshot,
        predictions: list[Prediction],
    ) -> CandidateDecision:
        if snapshot.length < self.config.minimum_frames:
            return CandidateDecision(None, 0.0, 0.0, "collecting")
        if snapshot.any_hand_coverage < self.config.minimum_hand_coverage:
            return CandidateDecision(None, 0.0, 0.0, "insufficient_hand_coverage")
        if snapshot.normalization_coverage < self.config.minimum_normalization_coverage:
            return CandidateDecision(None, 0.0, 0.0, "insufficient_pose_normalization")
        if len(predictions) < 2:
            raise ValueError("unknown rejection requires at least two ranked predictions")
        best, second = predictions[:2]
        margin = best.probability - second.probability
        if best.probability < self.config.minimum_probability:
            return CandidateDecision(None, best.probability, margin, "low_probability")
        if margin < self.config.minimum_margin:
            return CandidateDecision(None, best.probability, margin, "ambiguous")
        return CandidateDecision(best.label, best.probability, margin, "candidate")


@dataclass(frozen=True, slots=True)
class StabilityResult:
    candidate: str | None
    emitted: str | None
    consecutive: int
    reason: str


class TemporalStabilizer:
    """Require repeated candidates and suppress held-sign duplicates."""

    def __init__(
        self,
        required_consecutive: int = 3,
        cooldown_updates: int = 3,
        release_unknown_updates: int = 2,
    ) -> None:
        if required_consecutive <= 0:
            raise ValueError("required_consecutive must be positive")
        if cooldown_updates < 0 or release_unknown_updates <= 0:
            raise ValueError("cooldown must be non-negative and release count positive")
        self.required_consecutive = required_consecutive
        self.cooldown_updates = cooldown_updates
        self.release_unknown_updates = release_unknown_updates
        self.reset()

    def reset(self) -> None:
        self._candidate: str | None = None
        self._consecutive = 0
        self._unknown_updates = 0
        self._cooldown_remaining = 0
        self._last_emitted: str | None = None

    def update(self, decision: CandidateDecision) -> StabilityResult:
        if self._cooldown_remaining:
            self._cooldown_remaining -= 1

        if not decision.accepted:
            self._candidate = None
            self._consecutive = 0
            self._unknown_updates += 1
            if self._unknown_updates >= self.release_unknown_updates:
                self._last_emitted = None
            return StabilityResult(None, None, 0, decision.reason)

        self._unknown_updates = 0
        if decision.label == self._candidate:
            self._consecutive += 1
        else:
            self._candidate = decision.label
            self._consecutive = 1

        if self._consecutive < self.required_consecutive:
            return StabilityResult(
                self._candidate, None, self._consecutive, "stabilizing"
            )
        if self._cooldown_remaining:
            return StabilityResult(
                self._candidate, None, self._consecutive, "cooldown"
            )
        if self._candidate == self._last_emitted:
            return StabilityResult(
                self._candidate, None, self._consecutive, "duplicate_suppressed"
            )

        emitted = self._candidate
        self._last_emitted = emitted
        self._cooldown_remaining = self.cooldown_updates
        return StabilityResult(emitted, emitted, self._consecutive, "emitted")

