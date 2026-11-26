from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np

from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones a implementar por el estudiante:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)

    La orquestacion (search, index_image, persistencia y metricas de similitud)
    ya esta provista y no debe modificarse sin justificarlo en el Pull Request.
    """

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
        data_root: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver
        # DATA_PATH: las rutas de la galeria se guardan relativas a esta carpeta para que
        # la base indexada desde el entorno local sirva tambien en Docker (y viceversa).
        self.data_root = data_root
        # Backbone de extract_embedding: se carga en la primera llamada (ver _get_backbone).
        self._backbone = None
        self._device = None
        self._backbone_lock = threading.Lock()

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV)
        return image

    # ------------------------------------------------------------------
    # Etapa 1: funciones a implementar
    # ------------------------------------------------------------------

    def _get_backbone(self):
        """ResNet18 de ImageNet sin la capa fc, cargado una sola vez (thread-safe).

        Se eligio ResNet18 porque su embedding (512) coincide con EMBEDDING_DIM y con el del
        ResNet18 fine-tuned de la Etapa 2, asi la misma tabla de pgvector sirve para los dos y
        la comparacion baseline vs. fine-tuned es sobre la misma arquitectura.
        """
        if self._backbone is None:
            with self._backbone_lock:
                if self._backbone is None:
                    import torch
                    from torchvision.models import ResNet18_Weights, resnet18

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                    model.fc = torch.nn.Identity()
                    self._device = device
                    self._backbone = model.eval().to(device)
                    logger.info("Backbone baseline ResNet18 (ImageNet) cargado en %s", device)
        return self._backbone

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando un modelo pre-entrenado en
        ImageNet (ej: ResNet50, EfficientNet, ConvNeXt) sin la capa de
        clasificacion final.

        Implementacion: ResNet18 (ImageNet) con fc = Identity -> vector de 512 (salida del
        global average pooling), normalizado L2. El preprocesamiento es el compartido con el
        entrenamiento (lib.preprocessing): BGR -> RGB, letterbox a image_size, normalizacion
        de ImageNet y CHW.
        Retorna una lista de floats de dimension EMBEDDING_DIM.
        """
        import torch

        from lib.preprocessing import preprocess_bgr

        model = self._get_backbone()
        x = torch.from_numpy(preprocess_bgr(image, int(self.image_size))).unsqueeze(0)
        with torch.no_grad():
            emb = model(x.to(self._device)).squeeze(0).float().cpu().numpy()
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb.astype(float).tolist()

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Recupera de la base vectorial las top_k imagenes mas similares.

        - pgvector: self.store.search ordena por distancia coseno en la base. Para `l2` el
          orden es el mismo, porque todos los embeddings (consulta y galeria) estan
          normalizados L2 y ahi ||a - b|| = sqrt(2 - 2 cos(a, b)) es monotona en el coseno.
        - JSON: se recorre self.store.all().
        En ambos casos el score se calcula con self.similarity, que respeta
        SIMILARITY_METRIC (cosine | l2 -> 1 / (1 + distancia)).
        Retorna una lista de Neighbor (path, breed, score) ordenada por score
        descendente.
        """
        top_k = int(top_k)
        if top_k <= 0:
            return []
        store_search = getattr(self.store, "search", None)
        if callable(store_search):
            records = store_search(embedding, top_k)
        else:
            records = self.store.all()
        scored = [
            Neighbor(path=r.path, breed=r.breed, score=round(self.similarity(embedding, r.embedding), 6))
            for r in records
        ]
        scored.sort(key=lambda n: n.score, reverse=True)
        return scored[:top_k]

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Voto mayoritario ponderado por score entre las razas de los vecinos.

        Score devuelto: proporcion del voto ponderado que obtuvo la raza ganadora
        (suma de similitudes de sus vecinos / suma de similitudes de todos), en [0, 1].
        Mide el acuerdo entre vecinos: 1.0 si los K son de la misma raza.
        Empates: gana la raza del vecino mas cercano.

        Si no hay vecinos o el mejor score (similitud del vecino mas cercano) esta por
        debajo de self.similarity_threshold se considera "unknown" y se devuelve ese
        mejor score. El umbral se calibra con valid en 2-train.ipynb.
        Retorna (raza, score).
        """
        if not results:
            return "unknown", 0.0
        best = max(results, key=lambda n: n.score)
        if best.score < float(self.similarity_threshold):
            return "unknown", float(best.score)

        votes: dict[str, float] = {}
        for n in results:
            votes[n.breed] = votes.get(n.breed, 0.0) + max(float(n.score), 0.0)
        total = sum(votes.values())
        top = max(votes.values())
        winners = {b for b, v in votes.items() if v == top}
        if len(winners) == 1:
            breed = winners.pop()
        else:
            breed = next(n.breed for n in sorted(results, key=lambda n: n.score, reverse=True) if n.breed in winners)
        return breed, (votes[breed] / total if total > 0 else 0.0)

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def _to_stored_path(self, image_path: str) -> str:
        """Ruta relativa a DATA_PATH si la imagen esta dentro; si no, la ruta tal cual."""
        if self.data_root is None:
            return str(image_path)
        try:
            return Path(image_path).resolve().relative_to(self.data_root.resolve()).as_posix()
        except ValueError:
            return str(image_path)

    def resolve_path(self, stored_path: str) -> Path:
        """Inversa de _to_stored_path: ruta local de una imagen guardada en la base."""
        path = Path(stored_path)
        if self.data_root is None or path.is_absolute():
            return path
        candidate = self.data_root / path
        if not candidate.exists() and path.exists():
            # Base indexada antes de guardar rutas relativas a DATA_PATH.
            return path
        return candidate

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        """Extrae el embedding de una imagen del dataset y lo persiste en la base vectorial.

        La ruta se guarda relativa a DATA_PATH (ej: dataset/train/<raza>/<imagen>.jpg).
        Para leer la imagen a partir de un registro, usar resolve_path(record.path).
        """
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=self._to_stored_path(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(self.resolve_path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Pipeline completo de la Etapa 1: embedding -> vecinos -> raza predicha.

        `embedding_fn` permite seleccionar dinamicamente el extractor
        (baseline, resnet18_finetuned o cnn_custom, ver Etapa 2).
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        neighbors = [self._with_url(n) for n in self.search_similar_images(embedding, k)]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=model_name or self.model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
