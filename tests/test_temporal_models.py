import pytest
import torch

from isl_assistant.models.temporal import TemporalClassifier


@pytest.mark.parametrize("architecture", ["gru", "lstm", "transformer"])
def test_temporal_models_return_one_logit_vector_per_sequence(
    architecture: str,
) -> None:
    model = TemporalClassifier(
        architecture=architecture,
        input_size=255,
        hidden_size=32,
        number_of_classes=5,
        dropout=0.1,
        maximum_sequence_length=4,
        transformer_heads=4,
        transformer_layers=2,
        transformer_feedforward_size=64,
    )

    logits = model(torch.randn(3, 4, 255), torch.tensor([4, 3, 2]))

    assert logits.shape == (3, 5)


def test_transformer_rejects_incompatible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TemporalClassifier(
            architecture="transformer",
            input_size=255,
            hidden_size=30,
            number_of_classes=5,
            dropout=0.1,
            transformer_heads=4,
        )


def test_transformer_ignores_values_in_padded_frames() -> None:
    torch.manual_seed(3)
    model = TemporalClassifier(
        architecture="transformer",
        input_size=8,
        hidden_size=16,
        number_of_classes=3,
        dropout=0.0,
        maximum_sequence_length=4,
        transformer_heads=4,
        transformer_layers=2,
        transformer_feedforward_size=32,
    )
    model.eval()
    original = torch.randn(1, 4, 8)
    changed_padding = original.clone()
    changed_padding[:, 2:] = 1000.0

    with torch.no_grad():
        original_logits = model(original, torch.tensor([2]))
        changed_logits = model(changed_padding, torch.tensor([2]))

    torch.testing.assert_close(original_logits, changed_logits)
