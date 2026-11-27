"""Hiperparametros de entrenamiento por modelo (Etapa 2).

`train_classifier` usa estos valores por defecto; la notebook los sobreescribe para los
experimentos con `classifier.train_overrides = {...}` en lugar de editar el codigo.
Los valores finales se justifican en 2-train.ipynb.
"""
from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, dict[str, Any]] = {
    # Modelo A: fine-tuning completo con learning rates diferenciados (backbone 10x menor que la cabeza).
    "resnet18_finetuned": {
        "strategy": "finetune",       # finetune | feature_extraction
        "epochs": 15,
        "batch_size": 64,
        "lr": 1e-3,                   # cabeza
        "backbone_lr": 1e-4,          # backbone (solo con strategy=finetune)
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "dropout": 0.2,
        "warmup_epochs": 1,
        "patience": 5,                # early stopping sobre la accuracy de valid
        "augment": True,
        "seed": 42,
        "num_workers": 4,
        "amp": "auto",                # fp16 en GPU salvo GTX 16xx (NaN); True / False para forzar
    },
    # Modelo B: CNN propia desde cero; necesita mas epocas y regularizacion mas fuerte.
    "cnn_custom": {
        "epochs": 60,
        "batch_size": 64,
        "lr": 2e-3,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1,
        "dropout": 0.3,
        "warmup_epochs": 2,
        "patience": 12,
        "augment": True,
        "seed": 42,
        "num_workers": 4,
        "widths": (32, 64, 128, 256, 512),
        "embedding_dim": 512,
        "residual": True,
    },
}


def get_config(model_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if model_name not in DEFAULTS:
        raise ValueError(f"Sin configuracion de entrenamiento para '{model_name}'")
    cfg = dict(DEFAULTS[model_name])
    cfg.update(overrides or {})
    return cfg
