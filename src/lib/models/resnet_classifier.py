"""Modelo A: ResNet18 preentrenado en ImageNet con una cabeza nueva de una salida por raza."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetClassifier(nn.Module):
    """ResNet18 (ImageNet) + Dropout + Linear(512, n_clases).

    - `embed(x)`: salida del global average pooling (512), la capa previa al clasificador.
      Es el embedding que usa la busqueda por similitud con el modelo fine-tuned.
    - `forward(x)`: logits.
    - `classes`: lista de razas en el orden de los logits; viaja dentro del checkpoint.
    """

    def __init__(self, classes: list[str], dropout: float = 0.2, pretrained: bool = True) -> None:
        super().__init__()
        self.classes = list(classes)
        self.embedding_dim = 512
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, len(self.classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.embed(x)))

    def head_parameters(self):
        return self.classifier.parameters()

    def backbone_parameters(self):
        return self.backbone.parameters()
