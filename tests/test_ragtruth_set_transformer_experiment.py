from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from ragtruth_transfer.config import SetTransformerSettings
from ragtruth_transfer.modeling import HierarchicalEncoderClassifier
from ragtruth_transfer.ragtruth_confirmatory import ConfirmatoryConfig


ROOT = Path(__file__).resolve().parents[1]
NLLB_ATTENTION = ROOT / "configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml"
NLLB_SET = ROOT / "configs/ragtruth_pt_nllb_lora_set_transformer_confirmatory.yaml"
MADLAD_ATTENTION = ROOT / "configs/ragtruth_pt_madlad_lora_attention_mil_confirmatory.yaml"
MADLAD_SET = ROOT / "configs/ragtruth_pt_madlad_lora_set_transformer_confirmatory.yaml"
EN_SET = ROOT / "configs/ragtruth_lora_set_transformer_mil_confirmatory.yaml"


class ToyEncoder(nn.Module):
    """Deterministic encoder skeleton; no pretrained model or tokenizer is used."""

    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, token_type_ids, return_dict
        values = input_ids[:, 0].to(dtype=torch.float32).unsqueeze(-1)
        basis = torch.arange(self.hidden_size, dtype=torch.float32, device=input_ids.device)
        hidden = (values + basis).unsqueeze(1).expand(-1, input_ids.shape[1], -1) / 10.0
        return SimpleNamespace(last_hidden_state=hidden)


def _set_model(hidden_size: int = 8) -> HierarchicalEncoderClassifier:
    return HierarchicalEncoderClassifier(
        encoder=ToyEncoder(hidden_size),
        hidden_size=hidden_size,
        projection_size=128,
        dropout=0.0,
        architecture="set_transformer",
        attention_size=64,
        set_transformer=SetTransformerSettings(1, 4, 1, 128),
    )


def _bag(batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor(
        [[[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]]],
        dtype=torch.long,
    )
    return values.expand(batch_size, -1, -1).clone(), torch.ones((batch_size, 4), dtype=torch.bool)


def _projected(model: HierarchicalEncoderClassifier, input_ids: torch.Tensor) -> torch.Tensor:
    batch_size, n_evidence, sequence_length = input_ids.shape
    attention_mask = torch.ones_like(input_ids)
    outputs = model.encoder(
        input_ids.reshape(batch_size * n_evidence, sequence_length),
        attention_mask.reshape(batch_size * n_evidence, sequence_length),
        return_dict=True,
    )
    cls = outputs.last_hidden_state[:, 0, :]
    return model.projection(cls).reshape(batch_size, n_evidence, -1)


def _configs() -> tuple[ConfirmatoryConfig, ConfirmatoryConfig, ConfirmatoryConfig, ConfirmatoryConfig]:
    return tuple(ConfirmatoryConfig.from_yaml(path) for path in (NLLB_ATTENTION, NLLB_SET, MADLAD_ATTENTION, MADLAD_SET))  # type: ignore[return-value]


def _assert_shared_protocol(attention: ConfirmatoryConfig, set_transformer: ConfirmatoryConfig) -> None:
    left = attention.experiment
    right = set_transformer.experiment
    assert right.model_id == left.model_id
    assert right.model_revision == left.model_revision
    assert right.encoder_mode == left.encoder_mode
    assert right.max_length == left.max_length == 512
    assert right.truncation == left.truncation == "longest_first"
    assert right.projection_size == left.projection_size == 128
    assert right.attention_size == left.attention_size == 64
    assert right.dropout == left.dropout == 0.30
    assert right.gradient_checkpointing == left.gradient_checkpointing is True
    assert right.lora == left.lora
    assert right.training == left.training
    assert right.dataset == left.dataset
    assert set_transformer.seeds == attention.seeds == (0, 1, 2)
    assert set_transformer.split_seed == attention.split_seed == 42
    assert set_transformer.selection_metric == attention.selection_metric == "validation_AUPRC"
    assert set_transformer.selection_mode == attention.selection_mode == "max"
    assert set_transformer.experiment.output_root != attention.experiment.output_root


def test_existing_set_transformer_config_is_the_pt_backend_contract() -> None:
    en = ConfirmatoryConfig.from_yaml(EN_SET)
    _, nllb_set, _, _ = _configs()
    assert en.experiment.set_transformer == nllb_set.experiment.set_transformer
    assert en.experiment.set_transformer == SetTransformerSettings(1, 4, 1, 128)
    assert nllb_set.experiment.architecture == "set_transformer"
    assert nllb_set.experiment.canonical_pooling_type == "set_transformer"


def test_nllb_set_transformer_preserves_attention_protocol() -> None:
    nllb_attention, nllb_set, _, _ = _configs()
    _assert_shared_protocol(nllb_attention, nllb_set)
    assert nllb_attention.experiment.canonical_pooling_type == "attention"
    assert nllb_set.experiment.set_transformer == SetTransformerSettings(1, 4, 1, 128)
    assert nllb_set.experiment.lora.target_modules == ("query_proj", "value_proj")
    assert nllb_set.experiment.dataset.expected_signature == "ee8b7c597e51e120"
    assert nllb_set.expected_dataset_signature == "ee8b7c597e51e120"
    assert nllb_set.expected_split_signature == "611cd01cd5d41e18"


def test_madlad_matrix_is_prepared_without_fabricated_signatures() -> None:
    _, _, madlad_attention, madlad_set = _configs()
    _assert_shared_protocol(madlad_attention, madlad_set)
    assert madlad_attention.experiment.dataset.path == madlad_set.experiment.dataset.path
    assert madlad_attention.experiment.dataset.path == (ROOT / "data/processed/ragtruth_confirmatory_pt_madlad/dataset.parquet").resolve()
    assert madlad_attention.expected_dataset_signature is None
    assert madlad_set.expected_dataset_signature is None
    assert madlad_attention.expected_split_signature is None
    assert madlad_set.expected_split_signature is None
    assert madlad_attention.experiment.dataset.expected_signature is None
    assert madlad_set.experiment.dataset.expected_signature is None
    assert madlad_attention.experiment.output_root != madlad_set.experiment.output_root


def test_nllb_and_madlad_translation_matrix_preserves_model_protocol() -> None:
    nllb_attention, nllb_set, madlad_attention, madlad_set = _configs()
    for nllb, madlad in ((nllb_attention, madlad_attention), (nllb_set, madlad_set)):
        assert nllb.experiment.model_id == madlad.experiment.model_id
        assert nllb.experiment.model_revision == madlad.experiment.model_revision
        assert nllb.experiment.encoder_mode == madlad.experiment.encoder_mode
        assert nllb.experiment.lora == madlad.experiment.lora
        assert nllb.experiment.training == madlad.experiment.training
        assert nllb.experiment.max_length == madlad.experiment.max_length == 512
        assert nllb.experiment.truncation == madlad.experiment.truncation == "longest_first"
        assert nllb.experiment.dataset.validation_fraction == madlad.experiment.dataset.validation_fraction == 0.15
        assert nllb.experiment.dataset.split_seed == madlad.experiment.dataset.split_seed == 42
        assert nllb.experiment.dataset.group_column == madlad.experiment.dataset.group_column == "source_id"
        assert nllb.seeds == madlad.seeds == (0, 1, 2)
        assert nllb.experiment.dataset.path != madlad.experiment.dataset.path
        assert nllb.experiment.output_root != madlad.experiment.output_root
    assert nllb_attention.experiment.canonical_pooling_type == madlad_attention.experiment.canonical_pooling_type == "attention"
    assert nllb_set.experiment.canonical_pooling_type == madlad_set.experiment.canonical_pooling_type == "set_transformer"


def test_scheduler_horizon_is_unchanged_by_aggregator() -> None:
    nllb_attention, nllb_set, _, _ = _configs()
    train_examples = 25_347
    for config in (nllb_attention, nllb_set):
        steps_per_epoch = math.ceil(math.ceil(train_examples / config.experiment.training.train_batch_size) / config.experiment.training.gradient_accumulation_steps)
        assert steps_per_epoch == 1_585
        total_steps = steps_per_epoch * config.experiment.training.planned_total_epochs
        assert total_steps == 9_510
        assert int(total_steps * config.experiment.training.warmup_ratio) == 951


def test_set_transformer_forward_supports_batch_masks_and_normalized_weights() -> None:
    torch.manual_seed(100)
    model = _set_model()
    model.eval()
    input_ids, evidence_mask = _bag(batch_size=2)
    evidence_mask[1, 3] = False
    logits, weights = model(input_ids, torch.ones_like(input_ids), evidence_mask)
    assert logits.shape == (2,)
    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all(weights[1, 3] == 0)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(weights).any()


def test_masked_embedding_perturbation_does_not_change_set_output() -> None:
    torch.manual_seed(101)
    model = _set_model()
    model.eval()
    input_ids, evidence_mask = _bag()
    evidence_mask[0, 3] = False
    baseline_logits, baseline_weights = model(input_ids, torch.ones_like(input_ids), evidence_mask)
    perturbed = input_ids.clone()
    perturbed[0, 3, :] = 999
    perturbed_logits, perturbed_weights = model(perturbed, torch.ones_like(perturbed), evidence_mask)
    assert torch.allclose(baseline_logits, perturbed_logits, atol=1e-6)
    assert torch.allclose(baseline_weights, perturbed_weights, atol=1e-6)


def test_set_transformer_is_permutation_invariant_without_positional_encoding() -> None:
    torch.manual_seed(102)
    model = _set_model()
    model.eval()
    input_ids, evidence_mask = _bag()
    logits, weights = model(input_ids, torch.ones_like(input_ids), evidence_mask)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_ids = input_ids[:, permutation]
    permuted_mask = evidence_mask[:, permutation]
    permuted_logits, permuted_weights = model(permuted_ids, torch.ones_like(permuted_ids), permuted_mask)
    assert torch.allclose(logits, permuted_logits, atol=1e-6)
    assert torch.allclose(weights[:, permutation], permuted_weights, atol=1e-6)
    assert not any("position" in name.lower() for name, _ in model.named_modules())


def test_set_transformer_has_inter_evidence_interaction_and_gradients() -> None:
    torch.manual_seed(103)
    model = _set_model()
    input_ids, evidence_mask = _bag()
    projected = _projected(model, input_ids)
    contextualized = model.sab(projected, evidence_mask)
    changed = input_ids.clone()
    changed[0, 0, :] += 7
    changed_contextualized = model.sab(_projected(model, changed), evidence_mask)
    assert not torch.allclose(contextualized[0, 1], changed_contextualized[0, 1], atol=1e-6)
    logits, _ = model(input_ids, torch.ones_like(input_ids), evidence_mask)
    logits.sum().backward()
    assert model.sab.mab.attention.in_proj_weight.grad is not None
    assert model.pma.seed_vectors.grad is not None


def test_set_transformer_parameter_audit_matches_existing_head_contract() -> None:
    torch.manual_seed(104)
    attention = HierarchicalEncoderClassifier(
        encoder=ToyEncoder(768), hidden_size=768, projection_size=128, dropout=0.30,
        architecture="gated_attention", attention_size=64,
    )
    set_transformer = HierarchicalEncoderClassifier(
        encoder=ToyEncoder(768), hidden_size=768, projection_size=128, dropout=0.30,
        architecture="set_transformer", attention_size=64,
        set_transformer=SetTransformerSettings(1, 4, 1, 128),
    )
    attention_head = sum(parameter.numel() for parameter in attention.parameters())
    set_head = sum(parameter.numel() for parameter in set_transformer.parameters())
    assert attention_head == 115_137
    assert set_head == 297_857
    assert set_head - attention_head == 182_720
