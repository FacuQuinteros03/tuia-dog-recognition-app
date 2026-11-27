"""Modelo B: CNN propia entrenada desde cero.

Arquitectura (entrada 3 x 224 x 224):

    stem    conv 3x3 s2 (3 -> w0) + BN + ReLU, maxpool 2        -> w0 x 56 x 56
    bloque  [conv 3x3 + BN + ReLU] x 2 (+ atajo residual)  x 4  -> w4 x 7 x 7
            con maxpool 2 entre bloques (56 -> 28 -> 14 -> 7)
    GAP     global average pooling                               -> w4
    emb     Linear(w4, embedding_dim) + BN1d + ReLU              -> embedding_dim (= EMBEDDING_DIM)
    head    Dropout + Linear(embedding_dim, n_clases)            -> logits

Decisiones (se justifican con experimentos en 2-train.ipynb):
- El stem reduce a 56 x 56 antes de los bloques: casi todo el computo de una CNN esta en las
  capas de alta resolucion, y con 6000 imagenes no hace falta tanto detalle temprano.
- BatchNorm despues de cada convolucion: permite learning rates altos y entrenar desde cero sin
  que la red diverja.
- Atajo residual opcional (conv 1x1 cuando cambian los canales): facilita el flujo del gradiente.
- Global average pooling en lugar de aplanar: con 512 x 7 x 7 aplanado, la capa densa tendria
  ~12M de parametros y sobreajustaria con este dataset.
- Capa de embedding de dimension EMBEDDING_DIM antes del clasificador, para la busqueda.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, residual: bool = True) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.shortcut = None
        if residual:
            self.shortcut = (
                nn.Identity()
                if c_in == c_out
                else nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.act(out)


class DogCNN(nn.Module):
    """CNN propia. `embed(x)` devuelve la capa previa al clasificador; `classes` viaja en el checkpoint."""

    def __init__(
        self,
        classes: list[str],
        widths: tuple[int, ...] = (32, 64, 128, 256, 512),
        embedding_dim: int = 512,
        dropout: float = 0.3,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.classes = list(classes)
        self.embedding_dim = embedding_dim
        self.config = {"widths": list(widths), "embedding_dim": embedding_dim, "dropout": dropout, "residual": residual}
        w0, *rest = widths
        self.stem = nn.Sequential(
            nn.Conv2d(3, w0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        blocks: list[nn.Module] = []
        c_in = w0
        for i, c_out in enumerate(rest):
            blocks.append(ConvBlock(c_in, c_out, residual))
            if i < len(rest) - 1:
                blocks.append(nn.MaxPool2d(2))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(c_in, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, len(self.classes))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(self.stem(x))
        return self.embedding(torch.flatten(self.pool(x), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.embed(x)))

    def head_parameters(self):
        return self.classifier.parameters()
