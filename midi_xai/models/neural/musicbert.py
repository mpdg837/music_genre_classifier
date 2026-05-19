from collections.abc import Mapping

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class MusicBertGenreClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained_model_name_or_path: str | None = "manoskary/musicbert-large",
        encoder_config: Mapping[str, int | float | str | bool] | None = None,
        dropout: float = 0.1,
        classifier_hidden_dim: int | None = None,
        pooling: str = "mean",
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.pooling = pooling

        if pretrained_model_name_or_path is None:
            if encoder_config is None:
                raise ValueError(
                    "encoder_config is required when pretrained_model_name_or_path=None"
                )
            config = AutoConfig.for_model("bert", **dict(encoder_config))
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)

        hidden_size = int(self.encoder.config.hidden_size)
        if classifier_hidden_dim is None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, num_classes),
            )

        self.set_encoder_trainable(not freeze_encoder)

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(pooled)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return self._pool(outputs, attention_mask)

    def _pool(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0]

        if self.pooling == "pooler":
            if outputs.pooler_output is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state[:, 0]

        if self.pooling != "mean":
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

        mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return summed / lengths
