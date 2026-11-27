"""Preprocesamiento y data augmentation compartidos por las notebooks y los servicios de la app.

Un unico pipeline para entrenamiento e inferencia (ENUNCIADO 3.5):

    imagen BGR uint8 (cv2) -> RGB -> letterbox a size x size -> [0, 1] -> (x - mean) / std -> CHW float32

- Orden de canales: la app y cv2.imread entregan BGR; torchvision y los pesos de ImageNet esperan RGB.
- Letterbox: se escala el lado mayor a `size` y se completa con relleno hasta un cuadrado, sin
  deformar al perro ni recortarlo. El relleno es el color medio de ImageNet, que despues de
  normalizar queda en 0 (la misma senal que produce el padding de una convolucion). En el dataset
  todas las imagenes ya son 224x224, asi que ahi es una operacion identidad; importa para los
  recortes de YOLO (Etapa 3), que pueden ser muy alargados.
- Normalizacion: media y desvio de ImageNet para todos los modelos (ver 1-pre-processing.ipynb).

La augmentation (`build_train_augmentation`) se aplica solo en entrenamiento y antes de
`preprocess_rgb`, sobre la imagen RGB uint8.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
# Color de relleno del letterbox en RGB uint8: la media de ImageNet (queda en 0 al normalizar).
PAD_RGB: tuple[int, int, int] = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)  # type: ignore[assignment]


def read_bgr(path: str | Path) -> np.ndarray:
    """Lee una imagen como BGR uint8 (igual que cv2.imread), tambien con rutas no ASCII.

    Lanza ValueError si el archivo no existe o no se puede decodificar.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"No se pudo abrir la imagen: {path} ({exc})") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR -> RGB. Acepta imagenes en escala de grises (2D) y las pasa a 3 canales."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox(rgb: np.ndarray, size: int, fill: tuple[int, int, int] = PAD_RGB) -> np.ndarray:
    """Escala el lado mayor a `size` y centra la imagen en un cuadrado size x size relleno con `fill`."""
    h, w = rgb.shape[:2]
    if h == size and w == size:
        return rgb
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
    canvas = np.empty((size, size, 3), dtype=rgb.dtype)
    canvas[:] = fill
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def preprocess_rgb(
    rgb: np.ndarray,
    size: int,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """RGB uint8 (H, W, 3) -> tensor normalizado (3, size, size) float32."""
    x = letterbox(rgb, size).astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_bgr(
    image: np.ndarray,
    size: int,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Punto de entrada de los servicios: BGR uint8 (como llega a la app) -> (3, size, size) float32."""
    return preprocess_rgb(bgr_to_rgb(image), size, mean, std)


def build_train_augmentation(size: int) -> A.Compose:
    """Augmentation de entrenamiento sobre RGB uint8 (los rangos se justifican en la notebook 1).

    No incluye flips verticales, rotaciones de 90 grados ni cambios de tono que alteren el pelaje:
    no ocurren en fotos reales o cambian la raza aparente.
    """
    return A.Compose([
        # Encuadres mas ajustados o corridos, como los recortes de YOLO de la Etapa 3.
        A.RandomResizedCrop(size=(size, size), scale=(0.6, 1.0), ratio=(0.75, 1.333), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=(-0.05, 0.05),
                 border_mode=cv2.BORDER_CONSTANT, fill=PAD_RGB, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=(3, 5)),
        ], p=0.2),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
    ])
