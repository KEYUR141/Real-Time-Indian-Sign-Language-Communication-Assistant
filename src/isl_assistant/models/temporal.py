"""Compact recurrent and Transformer classifiers for landmark sequences."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalClassifier(nn.Module):
    """Classify padded sequences using a GRU, LSTM, or Transformer encoder."""

    def __init__(
        self,
        architecture: str,
        input_size: int,
        hidden_size: int,
        number_of_classes: int,
        dropout: float,
        *,
        maximum_sequence_length: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_feedforward_size: int = 256,
    ) -> None:
        super().__init__()
        if architecture not in {"gru", "lstm", "transformer"}:
            raise ValueError("architecture must be 'gru', 'lstm', or 'transformer'")
        self.architecture = architecture
        self.maximum_sequence_length = maximum_sequence_length
        if architecture == "transformer":
            if hidden_size % transformer_heads:
                raise ValueError("Transformer hidden size must be divisible by its head count")
            self.input_projection = nn.Linear(input_size, hidden_size)
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.position_embedding = nn.Parameter(
                torch.zeros(1, maximum_sequence_length + 1, hidden_size)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_heads,
                dim_feedforward=transformer_feedforward_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
                norm=nn.LayerNorm(hidden_size),
                enable_nested_tensor=False,
            )
            nn.init.normal_(self.class_token, std=0.02)
            nn.init.normal_(self.position_embedding, std=0.02)
        else:
            rnn_type = nn.GRU if architecture == "gru" else nn.LSTM
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
        if self.architecture == "transformer":
            return self._forward_transformer(sequences, lengths)
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

    def _forward_transformer(self, sequences: Tensor, lengths: Tensor) -> Tensor:
        batch_size, sequence_length, _ = sequences.shape
        if sequence_length > self.maximum_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.maximum_sequence_length}"
            )
        projected = self.input_projection(sequences)
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        encoded_input = torch.cat((class_tokens, projected), dim=1)
        encoded_input = encoded_input + self.position_embedding[:, : sequence_length + 1]
        device_lengths = lengths.to(sequences.device)
        padding = (
            torch.arange(sequence_length, device=sequences.device)[None, :]
            >= device_lengths[:, None]
        )
        class_padding = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=padding.device
        )
        padding = torch.cat((class_padding, padding), dim=1)
        encoded = self.transformer(
            encoded_input,
            src_key_padding_mask=padding,
        )
        return self.classifier(encoded[:, 0])
