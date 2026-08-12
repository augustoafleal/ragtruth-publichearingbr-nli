from pathlib import Path

import torch
import torch.nn as nn
import yaml

from ragtruth_transfer.config import ExperimentConfig
from ragtruth_transfer.ragtruth_confirmatory import ConfirmatoryConfig, _signature
from ragtruth_transfer.modeling import HierarchicalEncoderClassifier


class DummyOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.embedding = nn.Embedding(20, hidden_size)

    def forward(self, input_ids, attention_mask, return_dict=True, token_type_ids=None):
        return DummyOutput(self.embedding(input_ids))


class DummyLoRAEncoder(DummyEncoder):
    def __init__(self):
        super().__init__()
        self.lora_adapter = nn.Linear(4, 4, bias=False)

    def forward(self, input_ids, attention_mask, return_dict=True, token_type_ids=None):
        hidden = self.embedding(input_ids)
        return DummyOutput(hidden + self.lora_adapter(hidden))


class FixedClsEncoder(nn.Module):

    def forward(self, input_ids, attention_mask, return_dict=True, token_type_ids=None):
        hidden = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], 3),
            dtype=torch.float32,
            device=input_ids.device,
        )
        hidden[:, 0, :] = input_ids[:, :3].float()
        return DummyOutput(hidden)


def _deterministic_pooling_model(architecture: str) -> HierarchicalEncoderClassifier:
    model = HierarchicalEncoderClassifier(
        encoder=FixedClsEncoder(),
        hidden_size=3,
        projection_size=3,
        dropout=0.0,
        architecture=architecture,
        attention_size=2,
    )
    with torch.no_grad():
        model.projection[0].weight.copy_(torch.eye(3))
        model.projection[0].bias.zero_()
        model.classifier[1].weight.copy_(torch.tensor([[0.5, -0.25, 0.75]]))
        model.classifier[1].bias.copy_(torch.tensor([0.1]))
    return model


def _projected(model: HierarchicalEncoderClassifier, input_ids: torch.Tensor) -> torch.Tensor:
    batch_size, n_evidence, sequence_length = input_ids.shape
    outputs = model.encoder(
        input_ids=input_ids.reshape(batch_size * n_evidence, sequence_length),
        attention_mask=torch.ones_like(input_ids).reshape(batch_size * n_evidence, sequence_length),
        return_dict=True,
    )
    return model.projection(outputs.last_hidden_state[:, 0, :]).reshape(batch_size, n_evidence, -1)


def _pooled_before_classifier(
    model: HierarchicalEncoderClassifier,
    input_ids: torch.Tensor,
    evidence_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture(_module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = model.classifier.register_forward_pre_hook(capture)
    try:
        logits, weights = model(input_ids, torch.ones_like(input_ids), evidence_mask)
    finally:
        handle.remove()
    assert len(captured) == 1
    return captured[0], logits, weights


def test_all_pooling_architectures_respect_mask():
    for architecture in ["mean", "max", "gated_attention"]:
        model = HierarchicalEncoderClassifier(
            encoder=DummyEncoder(),
            hidden_size=4,
            projection_size=3,
            dropout=0.0,
            architecture=architecture,
            attention_size=2,
        )
        logits, weights = model(
            input_ids=torch.ones((2, 4, 3), dtype=torch.long),
            attention_mask=torch.ones((2, 4, 3), dtype=torch.long),
            evidence_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool),
        )
        assert logits.shape == (2,)
        assert weights.shape == (2, 4)
        assert torch.allclose(weights[0, 2:], torch.zeros(2), atol=1e-6)


def test_gated_attention_forward_backward_has_head_gradients():
    model = HierarchicalEncoderClassifier(
        encoder=DummyLoRAEncoder(), hidden_size=4, projection_size=3,
        dropout=0.0, architecture="gated_attention", attention_size=2,
    )
    logits, weights = model(
        input_ids=torch.ones((2, 4, 3), dtype=torch.long),
        attention_mask=torch.ones((2, 4, 3), dtype=torch.long),
        evidence_mask=torch.ones((2, 4), dtype=torch.bool),
    )
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0.0, 1.0])).backward()
    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert model.projection[0].weight.grad is not None
    assert model.classifier[1].weight.grad is not None
    assert model.encoder.lora_adapter.weight.grad is not None


def test_masked_slot_content_cannot_change_pooled_output():
    torch.manual_seed(3)
    model = HierarchicalEncoderClassifier(
        encoder=DummyEncoder(), hidden_size=4, projection_size=3,
        dropout=0.0, architecture="gated_attention", attention_size=2,
    )
    input_ids = torch.ones((1, 4, 3), dtype=torch.long)
    changed = input_ids.clone()
    changed[:, 2:, :] = 7
    mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)
    first, _ = model(input_ids, torch.ones_like(input_ids), mask)
    second, _ = model(changed, torch.ones_like(input_ids), mask)
    assert torch.allclose(first, second, atol=1e-6)


def test_mean_equals_manual_mean_over_only_valid_slots():
    model = _deterministic_pooling_model("mean")
    input_ids = torch.tensor([[
        [1, 2, 3], [4, 5, 6], [90, 90, 90], [7, 8, 9],
    ]])
    mask = torch.tensor([[True, False, False, True]])
    logits, weights = model(input_ids, torch.ones_like(input_ids), mask)
    projected = _projected(model, input_ids)
    pooled = projected[:, [0, 3], :].mean(dim=1)
    expected = model.classifier(pooled).squeeze(-1)
    assert torch.allclose(logits, expected, atol=1e-6)
    assert torch.allclose(weights, torch.tensor([[0.5, 0.0, 0.0, 0.5]]), atol=1e-6)


def test_mean_with_one_valid_slot_returns_that_projected_representation():
    model = _deterministic_pooling_model("mean")
    input_ids = torch.tensor([[
        [80, 80, 80], [1, 2, 3], [90, 90, 90], [70, 70, 70],
    ]])
    mask = torch.tensor([[False, True, False, False]])
    pooled, _logits, weights = _pooled_before_classifier(model, input_ids, mask)
    projected = _projected(model, input_ids)

    assert torch.equal(pooled, projected[:, 1, :])
    assert torch.equal(weights, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))


def test_max_equals_manual_featurewise_max_over_only_valid_slots():
    model = _deterministic_pooling_model("max")
    input_ids = torch.tensor([[
        [1, 7, 3], [80, 80, 80], [4, 2, 9], [70, 70, 70],
    ]])
    mask = torch.tensor([[True, False, True, False]])
    logits, weights = model(input_ids, torch.ones_like(input_ids), mask)
    projected = _projected(model, input_ids)
    pooled = torch.maximum(projected[:, 0, :], projected[:, 2, :])
    expected = model.classifier(pooled).squeeze(-1)
    assert torch.allclose(logits, expected, atol=1e-6)
    assert torch.allclose(weights[:, 1], torch.zeros(1), atol=1e-6)
    assert torch.allclose(weights[:, 3], torch.zeros(1), atol=1e-6)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-6)


def test_max_ignores_masked_zero_when_all_valid_features_are_negative():
    model = _deterministic_pooling_model("max")
    input_ids = torch.tensor([[
        [-3, -2, -1], [0, 0, 0], [-2, -1, -3], [0, 0, 0],
    ]])
    mask = torch.tensor([[True, False, True, False]])
    logits, _ = model(input_ids, torch.ones_like(input_ids), mask)
    projected = _projected(model, input_ids)
    assert bool((projected[:, [0, 2], :] < 0).all())
    pooled = torch.maximum(projected[:, 0, :], projected[:, 2, :])
    expected = model.classifier(pooled).squeeze(-1)
    assert torch.allclose(logits, expected, atol=1e-6)


def test_all_poolings_ignore_masked_slots_with_varied_valid_positions_and_preserve_shapes():
    input_ids = torch.tensor([
        [[1, 2, 3], [70, 70, 70], [80, 80, 80], [90, 90, 90]],
        [[1, 2, 3], [70, 70, 70], [4, 5, 6], [90, 90, 90]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [90, 90, 90]],
    ])
    changed = input_ids.clone()
    changed[~torch.tensor([
        [True, False, False, False],
        [True, False, True, False],
        [True, True, True, False],
    ])] = -99
    mask = torch.tensor([
        [True, False, False, False],
        [True, False, True, False],
        [True, True, True, False],
    ])
    for architecture in ("gated_attention", "mean", "max"):
        model = _deterministic_pooling_model(architecture)
        first_logits, first_weights = model(input_ids, torch.ones_like(input_ids), mask)
        second_logits, second_weights = model(changed, torch.ones_like(changed), mask)
        assert first_logits.shape == (3,)
        assert first_weights.shape == (3, 4)
        assert torch.allclose(first_logits, second_logits, atol=1e-6)
        assert torch.allclose(first_weights, second_weights, atol=1e-6)
        assert torch.allclose(first_weights[~mask], torch.zeros_like(first_weights[~mask]), atol=1e-6)


def test_all_poolings_produce_128_dimensional_representation_before_classifier():
    input_ids = torch.tensor([
        [[1, 2, 3], [70, 70, 70], [80, 80, 80], [90, 90, 90]],
        [[1, 2, 3], [70, 70, 70], [4, 5, 6], [90, 90, 90]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
    ])
    mask = torch.tensor([
        [True, False, False, False],
        [True, False, True, False],
        [True, True, True, True],
    ])
    for architecture in ("gated_attention", "mean", "max"):
        model = HierarchicalEncoderClassifier(
            encoder=FixedClsEncoder(),
            hidden_size=3,
            projection_size=128,
            dropout=0.0,
            architecture=architecture,
            attention_size=2,
        )
        pooled, logits, weights = _pooled_before_classifier(model, input_ids, mask)

        assert pooled.shape == (3, 128)
        assert logits.shape == (3,)
        assert weights.shape == (3, 4)


def test_confirmatory_pooling_configs_have_explicit_distinct_signatures():
    configs = [
        ConfirmatoryConfig.from_yaml(Path("configs/ragtruth_lora_attention_mil_confirmatory.yaml")),
        ConfirmatoryConfig.from_yaml(Path("configs/ragtruth_lora_mean_mil_confirmatory.yaml")),
        ConfirmatoryConfig.from_yaml(Path("configs/ragtruth_lora_max_mil_confirmatory.yaml")),
    ]
    assert [config.experiment.canonical_pooling_type for config in configs] == ["attention", "mean", "max"]
    assert configs[1].experiment.to_dict()["pooling_type"] == "mean"
    assert configs[2].experiment.to_dict()["pooling_type"] == "max"
    signatures = {
        _signature(config.protocol_payload("525edec2966a4fac"))
        for config in configs
    }
    assert len(signatures) == 3


def test_attention_alias_normalizes_to_legacy_gated_attention():
    config = ExperimentConfig.from_mapping(
        {"run_name": "alias", "model_id": "dummy", "architecture": "attention", "training": {}}
    )
    assert config.architecture == "gated_attention"
    assert config.canonical_pooling_type == "attention"
    assert config.to_dict()["pooling_type"] == "attention"


def test_mean_and_max_confirmatory_configs_change_only_pooling_identity():
    base = yaml.safe_load(Path("configs/ragtruth_lora_attention_mil_confirmatory.yaml").read_text())
    base.pop("run_name")
    base.pop("architecture")
    for name, architecture in (("mean", "mean"), ("max", "max")):
        variant = yaml.safe_load(Path(f"configs/ragtruth_lora_{name}_mil_confirmatory.yaml").read_text())
        assert variant.pop("run_name") == f"ragtruth_lora_{name}_mil_confirmatory"
        assert variant.pop("architecture") == architecture
        assert variant.pop("pooling_type") == architecture
        assert variant == base
