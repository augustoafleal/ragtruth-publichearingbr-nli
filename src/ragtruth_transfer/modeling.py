from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import ExperimentConfig, SetTransformerSettings


class MAB(nn.Module):
    """Multihead Attention Block used by the Set Transformer.

    This is the MAB structure from Set Transformer: multi-head attention,
    residual connection and LayerNorm, followed by a residual FFN and a second
    LayerNorm.  It intentionally has no positional encoding.
    """

    def __init__(self, dimension: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dimension,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm_attention = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dimension),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(dimension)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        hidden = self.norm_attention(query + self.attention_dropout(attended))
        output = self.norm_ffn(hidden + self.ffn_dropout(self.ffn(hidden)))
        return output, weights


class SAB(nn.Module):
    """Set Attention Block: MAB(X, X)."""

    def __init__(self, dimension: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.mab = MAB(dimension, num_heads, ffn_dim, dropout)

    def forward(self, values: torch.Tensor, evidence_mask: torch.Tensor) -> torch.Tensor:
        output, _ = self.mab(
            values,
            values,
            ~evidence_mask,
        )
        # Invalid slots are never keys/values and are made inert before PMA.
        return output.masked_fill(~evidence_mask.unsqueeze(-1), 0.0)


class PMA(nn.Module):
    """Pooling by Multihead Attention with learned seed vectors."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        ffn_dim: int,
        num_seeds: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.empty(1, num_seeds, dimension))
        nn.init.xavier_uniform_(self.seed_vectors)
        self.mab = MAB(dimension, num_heads, ffn_dim, dropout)

    def forward(
        self,
        values: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = values.shape[0]
        query = self.seed_vectors.expand(batch_size, -1, -1)
        output, weights = self.mab(
            query,
            values,
            ~evidence_mask,
            need_weights=True,
        )
        if weights is None:
            raise RuntimeError("PMA deve retornar pesos de atenção")
        # PyTorch returns head-averaged weights because average_attn_weights is
        # true: [B, num_seeds, 4].  They are diagnostics, not Gated weights.
        weights = weights[:, 0, :].masked_fill(~evidence_mask, 0.0)
        normalizer = weights.sum(dim=1, keepdim=True)
        fallback = evidence_mask.to(values.dtype)
        fallback = fallback / fallback.sum(dim=1, keepdim=True)
        normalized = weights / normalizer.clamp_min(torch.finfo(weights.dtype).eps)
        weights = torch.where(normalizer > 0, normalized, fallback)
        return output, weights


class HierarchicalEncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        projection_size: int,
        dropout: float,
        architecture: str,
        attention_size: int,
        set_transformer: SetTransformerSettings | None = None,
    ) -> None:
        super().__init__()
        if architecture not in {"mean", "max", "gated_attention", "set_transformer"}:
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
        elif architecture == "set_transformer":
            if set_transformer is None:
                raise ValueError("set_transformer requer parâmetros explícitos")
            if projection_size != set_transformer.ffn_dim:
                raise ValueError("Set Transformer requer ffn_dim igual a projection_size")
            self.sab = SAB(
                projection_size,
                set_transformer.num_heads,
                set_transformer.ffn_dim,
                dropout,
            )
            self.pma = PMA(
                projection_size,
                set_transformer.num_heads,
                set_transformer.ffn_dim,
                set_transformer.num_seeds,
                dropout,
            )
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
        elif self.architecture == "gated_attention":
            logits = self.attention_w(
                self.attention_v(projected) * self.attention_u(projected)
            ).squeeze(-1)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=1)
            pooled = torch.sum(projected * weights.unsqueeze(-1), dim=1)
        else:
            contextualized = self.sab(projected, mask)
            pooled, weights = self.pma(contextualized, mask)
            pooled = pooled.squeeze(1)

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
    if config.set_transformer is not None:
        metadata["set_transformer"] = {
            "num_sab_layers": config.set_transformer.num_sab_layers,
            "num_heads": config.set_transformer.num_heads,
            "num_seeds": config.set_transformer.num_seeds,
            "ffn_dim": config.set_transformer.ffn_dim,
            "positional_encoding": False,
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
        set_transformer=config.set_transformer,
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
