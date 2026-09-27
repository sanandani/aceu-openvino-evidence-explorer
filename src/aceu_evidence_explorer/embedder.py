from __future__ import annotations

import platform
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_DEVICE, DEFAULT_MODEL


class OpenVINOEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = DEFAULT_DEVICE):
        self.model_name = model_name
        self.device = str(device).strip().lower() if device else "cpu"
        self.model: SentenceTransformer | None = None
        self.backend_name = "openvino"
        self._initialize()

    def _initialize(self) -> None:
        try:
            self.model = SentenceTransformer(self.model_name, backend="openvino", device=self.device)
        except Exception as exc:
            raise RuntimeError(
                "OpenVINO initialization failed. Install the project dependencies from requirements.txt and ensure the model can be downloaded or is already cached."
            ) from exc

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            self._initialize()
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def similarity(self, query: str, candidates: Sequence[str]) -> np.ndarray:
        if not candidates:
            return np.empty((0,), dtype=np.float32)
        query_embedding = self.encode([query])[0]
        candidate_embeddings = self.encode(candidates)
        query_norm = np.linalg.norm(query_embedding)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
        query_embedding = query_embedding / max(query_norm, 1e-9)
        candidate_embeddings = candidate_embeddings / np.maximum(candidate_norms[:, None], 1e-9)
        return candidate_embeddings @ query_embedding

    @property
    def info(self) -> dict[str, str]:
        return {
            "model_id": self.model_name,
            "backend": self.backend_name,
            "device": self.device,
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
