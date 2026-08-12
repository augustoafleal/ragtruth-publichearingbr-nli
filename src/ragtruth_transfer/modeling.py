from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import ExperimentConfig


class HierarchicalEncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        projection_size: int,
        dropout: float,
        architecture: str,
        attention_size: int,
    ) -> None:
        super().__init__()
        if architecture not in {"mean", "max", "gated_attention"}:
            raise ValueError(architecture)
        self.encoder = encoder
        self.architecture = architecture
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if architecture == "gated_attention":
            self.attention_v = nn.Sequential(nn.Linear(projection_size, attention_size), nn.Tanh())
            self.attention_u = nn.Sequential(nn.Linear(projection_size, attention_size), nn.Sigmoid())
            self.attention_w = nn.Linear(attention_size, 1, bias=False)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(projection_size, 1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_evidence, sequence_length = input_ids.shape
        flat_kwargs: dict[str, torch.Tensor | bool] = {
            "input_ids": input_ids.reshape(batch_size * n_evidence, sequence_length),
            "attention_mask": attention_mask.reshape(batch_size * n_evidence, sequence_length),
            "return_dict": True,
        }
        if token_type_ids is not None:
            flat_kwargs["token_type_ids"] = token_type_ids.reshape(
                batch_size * n_evidence, sequence_length
            )
        outputs = self.encoder(**flat_kwargs)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = cls.to(dtype=self.projection[0].weight.dtype)
        projected = self.projection(cls).reshape(batch_size, n_evidence, -1)
        mask = evidence_mask.bool()
        if not torch.all(mask.any(dim=1)):
            raise RuntimeError("Toda bag precisa conter pelo menos uma evidência válida.")

        if self.architecture == "mean":
            weights = mask.to(projected.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)
            pooled = torch.sum(projected * weights.unsqueeze(-1), dim=1)
        elif self.architecture == "max":
            masked = projected.masked_fill(~mask.unsqueeze(-1), torch.finfo(projected.dtype).min)
            pooled = masked.max(dim=1).values
            maxima = masked.argmax(dim=1)
            weights = torch.zeros_like(mask, dtype=projected.dtype)
            for evidence_index in range(n_evidence):
                weights[:, evidence_index] = (maxima == evidence_index).to(projected.dtype).mean(dim=1)
        else:
            logits = self.attention_w(
                self.attention_v(projected) * self.attention_u(projected)
            ).squeeze(-1)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=1)
            pooled = torch.sum(projected * weights.unsqueeze(-1), dim=1)

        logits = self.classifier(pooled).squeeze(-1)
        return logits, weights


def _validate_lora_targets(encoder: nn.Module, targets: tuple[str, ...]) -> dict[str, list[str]]:
    names = [name for name, _ in encoder.named_modules()]
    matches = {target: [name for name in names if name.endswith(target)] for target in targets}
    missing = [target for target, values in matches.items() if not values]
    if missing:
        raise RuntimeError(f"Módulos LoRA ausentes: {missing}. Correspondências: {matches}")
    return matches


def build_model(
    config: ExperimentConfig,
    adapter_path: Path | None = None,
    adapter_trainable: bool = False,
) -> tuple[HierarchicalEncoderClassifier, dict[str, Any]]:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModel

    model_config = AutoConfig.from_pretrained(
        config.model_id,
        revision=config.model_revision,
    )
    encoder = AutoModel.from_pretrained(
        config.model_id,
        revision=config.model_revision,
    )
    metadata: dict[str, Any] = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "architecture": config.architecture,
        "pooling_type": config.canonical_pooling_type,
        "encoder_mode": config.encoder_mode,
        "hidden_size": int(model_config.hidden_size),
    }

    if config.encoder_mode == "lora":
        matches = _validate_lora_targets(encoder, config.lora.target_modules)
        if adapter_path is None:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=list(config.lora.target_modules),
                bias="none",
            )
            encoder = get_peft_model(encoder, lora_config)
        else:
            encoder = PeftModel.from_pretrained(
                encoder, adapter_path, is_trainable=adapter_trainable
            )
        metadata["lora_matches"] = matches
    else:
        for parameter in encoder.parameters():
            parameter.requires_grad = False

    if config.gradient_checkpointing and (adapter_path is None or adapter_trainable) and hasattr(encoder, "gradient_checkpointing_enable"):
        encoder.gradient_checkpointing_enable()
        if hasattr(encoder, "enable_input_require_grads"):
            encoder.enable_input_require_grads()
        metadata["gradient_checkpointing"] = True

    model = HierarchicalEncoderClassifier(
        encoder=encoder,
        hidden_size=int(model_config.hidden_size),
        projection_size=config.projection_size,
        dropout=config.dropout,
        architecture=config.architecture,
        attention_size=config.attention_size,
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    metadata["trainable_parameters"] = int(trainable)
    metadata["total_parameters"] = int(total)
    metadata["trainable_fraction"] = float(trainable / total)
    return model, metadata


def head_state_dict(model: HierarchicalEncoderClassifier) -> dict[str, torch.Tensor]:
    encoder_prefix = "encoder."
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(encoder_prefix)
    }


def load_head_state(model: HierarchicalEncoderClassifier, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_non_encoder = [key for key in missing if not key.startswith("encoder.")]
    if missing_non_encoder or unexpected:
        raise RuntimeError(
            f"Estado da cabeça incompatível. missing={missing_non_encoder}, unexpected={unexpected}"
        )
