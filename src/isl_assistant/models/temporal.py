"""Compact GRU and LSTM classifiers for landmark sequences."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalClassifier(nn.Module):
    """Classify padded landmark sequences using a GRU or LSTM encoder."""

    def __init__(
        self,
        architecture: str,
        input_size: int,
        hidden_size: int,
        number_of_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if architecture not in {"gru", "lstm"}:
            raise ValueError("architecture must be 'gru' or 'lstm'")
        rnn_type = nn.GRU if architecture == "gru" else nn.LSTM
        self.architecture = architecture
        self.rnn = rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, number_of_classes),
        )

    def forward(self, sequences: Tensor, lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(
            sequences,
            lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)
        if self.architecture == "lstm":
            hidden = hidden[0]
        return self.classifier(hidden[-1])
