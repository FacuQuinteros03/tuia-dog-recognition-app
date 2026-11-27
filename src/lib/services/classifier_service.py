from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

logger = logging.getLogger(__name__)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (resnet18_finetuned): ResNet18 de ImageNet con cabeza nueva; estrategia
        `finetune` (todo el modelo, learning rate del backbone 10x menor) o `feature_extraction`
        (backbone congelado). Modelo B (cnn_custom): CNN propia desde cero (lib/models/cnn_custom.py).

        - Splits train/valid de 1-pre-processing.ipynb (carpetas de self.dataset_path).
        - Preprocesamiento y augmentation compartidos (lib/preprocessing.py).
        - Hiperparametros en lib/models/train_config.py; la notebook los puede sobreescribir con
          `self.train_overrides = {...}` (incluye `checkpoint` y `run_name` para experimentos).
        - AdamW, warmup lineal + coseno por paso, label smoothing, precision mixta en GPU.
        - Early stopping: se guarda el modelo de la epoca con mejor accuracy en valid.
        - Checkpoint: modelo completo (torch.save) con `classes` como atributo, en
          self.active_checkpoint. Historial por epoca en output_path/history_<run_name>.json.
        """
        import copy
        import json
        import math
        import random
        import time

        from torch.utils.data import DataLoader

        from lib.models.data import DogImageDataset, list_classes, list_split, seed_worker
        from lib.models.train_config import get_config

        name = self.active_model_name
        cfg = get_config(name, getattr(self, "train_overrides", None))
        run_name = cfg.get("run_name", name)
        seed = int(cfg["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Precision mixta (fp16) solo si la GPU la soporta bien: en las GTX 16xx (Turing sin tensor
        # cores) el forward en fp16 del ResNet da NaN, asi que ahi se entrena en fp32.
        amp_cfg = cfg.get("amp", "auto")
        if amp_cfg == "auto":
            use_amp = device.type == "cuda" and "GTX 16" not in torch.cuda.get_device_name(0)
        else:
            use_amp = bool(amp_cfg) and device.type == "cuda"
        torch.backends.cudnn.benchmark = True

        classes = list_classes(self.dataset_path)
        tr_paths, tr_labels = list_split(self.dataset_path, "train", classes)
        va_paths, va_labels = list_split(self.dataset_path, "valid", classes)
        workers = int(cfg["num_workers"])
        loader_kw = {"num_workers": workers, "pin_memory": use_amp, "persistent_workers": workers > 0}
        train_loader = DataLoader(
            DogImageDataset(tr_paths, tr_labels, self.image_size, augment=bool(cfg["augment"])),
            batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True,
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed), **loader_kw,
        )
        valid_loader = DataLoader(
            DogImageDataset(va_paths, va_labels, self.image_size, augment=False),
            batch_size=int(cfg["batch_size"]), shuffle=False, **loader_kw,
        )

        model = self._build_model(name, classes, cfg).to(device)
        feature_extraction = cfg.get("strategy") == "feature_extraction"
        if name == "resnet18_finetuned":
            if feature_extraction:
                for p in model.backbone_parameters():
                    p.requires_grad = False
                groups = [{"params": list(model.head_parameters()), "lr": cfg["lr"]}]
            else:
                groups = [
                    {"params": list(model.backbone_parameters()), "lr": cfg["backbone_lr"]},
                    {"params": list(model.head_parameters()), "lr": cfg["lr"]},
                ]
        else:
            groups = [{"params": list(model.parameters()), "lr": cfg["lr"]}]
        optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg["weight_decay"]))

        epochs = int(cfg["epochs"])
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch
        warmup_steps = int(cfg["warmup_epochs"]) * steps_per_epoch

        def lr_factor(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{run_name}] {n_params:,} parametros ({n_train:,} entrenables) | {len(tr_paths)} train, "
              f"{len(va_paths)} valid | {steps_per_epoch} pasos/epoca | {device} | amp={use_amp}")

        history: list[dict[str, float]] = []
        best_acc, best_loss, best_epoch, best_state = -1.0, float("inf"), 0, None
        bad_epochs = 0
        t_start = time.time()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            if feature_extraction:
                model.backbone.eval()  # BatchNorm del backbone congelado: no actualiza estadisticas
            tr_loss = tr_correct = tr_n = 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                tr_loss += loss.item() * len(y)
                tr_correct += int((logits.argmax(1) == y).sum())
                tr_n += len(y)
            va_loss, va_acc = self._validate(model, valid_loader, criterion, device)
            row = {
                "epoch": epoch, "train_loss": tr_loss / tr_n, "train_acc": tr_correct / tr_n,
                "val_loss": va_loss, "val_acc": va_acc, "lr": optimizer.param_groups[-1]["lr"],
                "seconds": time.time() - t0,
            }
            history.append(row)
            improved = va_acc > best_acc or (va_acc == best_acc and va_loss < best_loss)
            if improved:
                best_acc, best_loss, best_epoch = va_acc, va_loss, epoch
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                bad_epochs = 0
            else:
                bad_epochs += 1
            print(f"[{run_name}] epoca {epoch:3d}/{epochs} | train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}"
                  f" | valid loss {va_loss:.3f} acc {va_acc:.3f}{' *' if improved else ''} | {row['seconds']:.0f}s")
            if bad_epochs >= int(cfg["patience"]):
                print(f"[{run_name}] early stopping: {bad_epochs} epocas sin mejorar en valid")
                break

        model.load_state_dict(best_state)
        model = model.cpu().eval()
        for p in model.parameters():
            p.requires_grad = True
        serializable_cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        model.train_config = serializable_cfg
        model.best_epoch = best_epoch
        model.image_size = self.image_size

        checkpoint = Path(cfg.get("checkpoint") or self.active_checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, checkpoint)
        self._loaded.pop(name, None)  # la proxima carga lee el checkpoint nuevo
        getattr(self, "_on_device", {}).pop(name, None)

        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / f"history_{run_name}.json").write_text(json.dumps({
            "model": name, "run_name": run_name, "config": serializable_cfg, "checkpoint": str(checkpoint),
            "params": n_params, "trainable_params": n_train, "best_epoch": best_epoch,
            "best_val_acc": best_acc, "train_seconds": time.time() - t_start, "history": history,
        }, indent=2), encoding="utf-8")
        print(f"[{run_name}] mejor epoca {best_epoch} (valid acc {best_acc:.3f}) -> {checkpoint}")

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba (carpeta test de self.dataset_path).

        Retorna accuracy, precision, recall (sensibilidad), specificity (especificidad) y F1.
        Con 70 clases, precision / recall / F1 son macro (promedio por clase) y la especificidad
        se calcula one-vs-rest por clase (TN / (TN + FP)) y se promedia. Las predicciones se
        guardan en output_path/eval_<modelo>_test.npz para la matriz de confusion de la notebook.
        """
        from lib.models.data import list_split

        model, _ = self._torch_model()
        paths, labels = list_split(self.dataset_path, "test", model.classes)
        probs, _ = self._infer_paths(paths)
        y_true, y_pred = np.asarray(labels), probs.argmax(1)
        metrics = self._summary_metrics(y_true, y_pred, len(model.classes))
        self.output_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.output_path / f"eval_{self.active_model_name}_test.npz",
            paths=np.asarray(paths), y_true=y_true, probs=probs.astype(np.float32),
            classes=np.asarray(model.classes),
        )
        return metrics

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Embedding de la capa previa al clasificador del modelo activo, normalizado L2.

        - resnet18_finetuned: salida del global average pooling del ResNet18 fine-tuned (512).
        - cnn_custom: capa de embedding de la CNN propia (EMBEDDING_DIM = 512).
        Mismo preprocesamiento que el entrenamiento (lib.preprocessing.preprocess_bgr).
        La imagen llega en BGR (OpenCV).
        """
        from lib.preprocessing import preprocess_bgr

        model, device = self._torch_model()
        x = torch.from_numpy(preprocess_bgr(image, self.image_size)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.embed(x).float().cpu().numpy()[0]
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb.astype(float).tolist()

    # ------------------------------------------------------------------
    # Helpers de la Etapa 2 (usados por la notebook y por la Etapa 3)
    # ------------------------------------------------------------------

    def predict_breed(self, image: np.ndarray) -> tuple[str, float]:
        """Clasifica una imagen BGR con el modelo activo: (raza, probabilidad softmax)."""
        from lib.preprocessing import preprocess_bgr

        model, device = self._torch_model()
        x = torch.from_numpy(preprocess_bgr(image, self.image_size)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x).float(), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return model.classes[idx], float(probs[idx])

    @staticmethod
    def _build_model(name: str, classes: list[str], cfg: dict[str, Any]) -> torch.nn.Module:
        if name == "resnet18_finetuned":
            from lib.models.resnet_classifier import ResNetClassifier

            return ResNetClassifier(classes, dropout=float(cfg["dropout"]))
        from lib.models.cnn_custom import DogCNN

        return DogCNN(classes, widths=tuple(cfg["widths"]), embedding_dim=int(cfg["embedding_dim"]),
                      dropout=float(cfg["dropout"]), residual=bool(cfg["residual"]))

    def _torch_model(self) -> tuple[torch.nn.Module, torch.device]:
        """Modelo activo (.pth) en modo eval, en GPU si hay; se mueve una sola vez."""
        model = self.load_model()
        if isinstance(model, onnxruntime.InferenceSession):
            raise ValueError("Este servicio usa checkpoints .pth (modelo completo con embed() y classes)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(self, "_on_device"):
            self._on_device: dict[str, Any] = {}
        if self._on_device.get(self.active_model_name) is not model:
            model.to(device).eval()
            self._on_device[self.active_model_name] = model
        return model, device

    @staticmethod
    def _validate(model, loader, criterion, device) -> tuple[float, float]:
        model.eval()
        loss_sum = correct = n = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss_sum += loss.item() * len(y)
                correct += int((logits.argmax(1) == y).sum())
                n += len(y)
        return loss_sum / n, correct / n

    def _infer_paths(self, paths: list[str], batch_size: int = 64, num_workers: int = 2):
        """Probabilidades (N, clases) y embeddings normalizados L2 (N, dim) del modelo activo, por lotes."""
        from torch.utils.data import DataLoader

        from lib.models.data import DogImageDataset

        model, device = self._torch_model()
        loader = DataLoader(DogImageDataset(paths, [0] * len(paths), self.image_size), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
        probs, embs = [], []
        with torch.no_grad():
            for x, _ in loader:
                e = model.embed(x.to(device))
                probs.append(torch.softmax(model.classifier(e).float(), 1).cpu().numpy())
                embs.append(torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy())
        return np.concatenate(probs), np.concatenate(embs)

    @staticmethod
    def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, np.ndarray]:
        """Precision, recall, F1 y especificidad one-vs-rest por clase (con metrics.py)."""
        from lib.evaluation.metrics import precision_recall_f1, specificity

        cm = np.bincount(n_classes * np.asarray(y_true) + np.asarray(y_pred), minlength=n_classes ** 2).reshape(n_classes, n_classes)
        total = cm.sum()
        out = {k: np.zeros(n_classes) for k in ("precision", "recall", "f1", "specificity", "support")}
        for c in range(n_classes):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum()) - tp
            fn = int(cm[c, :].sum()) - tp
            tn = int(total) - tp - fp - fn
            out["precision"][c], out["recall"][c], out["f1"][c] = precision_recall_f1(tp, fp, fn)
            out["specificity"][c] = specificity(tn, fp)
            out["support"][c] = tp + fn
        out["confusion"] = cm
        return out

    @classmethod
    def _summary_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
        pc = cls._per_class_metrics(y_true, y_pred, n_classes)
        present = pc["support"] > 0
        return {
            "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
            "precision": float(pc["precision"][present].mean()),
            "recall": float(pc["recall"][present].mean()),
            "specificity": float(pc["specificity"][present].mean()),
            "f1": float(pc["f1"][present].mean()),
        }
