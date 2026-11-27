from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones a implementar por el estudiante:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV / ultralytics)
        return image

    # ------------------------------------------------------------------
    # Etapa 3: funciones a implementar
    # ------------------------------------------------------------------

    def _get_yolo(self):
        """Detector YOLO preentrenado en COCO, cargado una sola vez (thread-safe).

        ultralytics descarga los pesos en la ruta de YOLO_MODEL (dentro de models/) en el primer uso.
        """
        import threading

        if not hasattr(self, "_yolo_lock"):
            self._yolo_lock = threading.Lock()
        if getattr(self, "_yolo", None) is None:
            with self._yolo_lock:
                if getattr(self, "_yolo", None) is None:
                    from ultralytics import YOLO

                    self._yolo = YOLO(self.yolo_model_name)
                    logger.info("Detector YOLO cargado: %s", self.yolo_model_name)
        return self._yolo

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Detecta todos los perros de la imagen con YOLO preentrenado en COCO (YOLO_MODEL).

        Se filtra por la clase perro (YOLO_DOG_CLASS_ID = 16) y por confianza
        (YOLO_CONF_THRESHOLD), ambos del .env. La imagen llega en BGR, que es lo que espera
        ultralytics para arrays de numpy. Las cajas se devuelven en pixeles enteros de la imagen
        original, ordenadas por confianza descendente; sin perros, lista vacia.

        Retorna una lista de ((x1, y1, x2, y2), confidence) en pixeles.
        """
        if image is None or image.size == 0:
            return []
        result = self._get_yolo().predict(
            image,
            conf=float(self.conf_threshold),
            classes=[int(self.dog_class_id)],
            device=getattr(self, "device", None),
            verbose=False,
        )[0]
        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()
        ):
            if int(cls) != int(self.dog_class_id) or float(conf) < float(self.conf_threshold):
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in box)
            detections.append(((x1, y1, x2, y2), float(conf)))
        detections.sort(key=lambda d: d[1], reverse=True)
        return detections

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro con el modelo activo de la Etapa 2.

        Usa ClassifierService.predict_breed: el mismo preprocesamiento del entrenamiento
        (BGR -> RGB, letterbox a IMAGE_SIZE sin deformar recortes alargados, normalizacion de
        ImageNet), model.eval() + no_grad, y la lista de razas guardada dentro del checkpoint.
        El score es la probabilidad softmax de la raza predicha. Un recorte vacio da "unknown".

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 2:
            return "unknown", 0.0
        return self.classifier.predict_breed(crop)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2).

        Reutiliza classify_detected_dog tratando la imagen entera como recorte,
        por lo que requiere la Etapa 2 (modelo entrenado) y classify_detected_dog.
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        # El modelo elegido vale solo para esta clasificacion: se restaura el activo para
        # que /detect siga usando el modelo configurado.
        previous = self.classifier.active_model_name
        if model_name:
            self.classifier.set_active_model(model_name)
        try:
            breed, score = self.classify_detected_dog(image)
        finally:
            self.classifier.set_active_model(previous)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion.

        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
