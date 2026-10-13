from __future__ import annotations

import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_MODEL

SAMPLE_INPUTS = [
    "We need a public-interest systems engineer with mentorship and reliability experience.",
    "We need a human-centered AI review lead focused on evaluation design and uncertainty communication.",
    "We need someone who can troubleshoot edge deployments and support operational teams.",
    "A community technology organizer documented accessible workshops and outcomes.",
]
BATCH_SIZES = (1, 2, 4)
REPEATS = 30
WARMUP = 5


def _versions() -> dict[str, str]:
    names = ("torch", "sentence-transformers", "openvino", "optimum-intel", "numpy")
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


def _run(model: SentenceTransformer, texts: list[str], repeats: int) -> tuple[dict[str, float], np.ndarray]:
    for _ in range(WARMUP):
        model.encode(texts, batch_size=len(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    timings = []
    embeddings = None
    for _ in range(repeats):
        start = time.perf_counter()
        embeddings = model.encode(texts, batch_size=len(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        timings.append(time.perf_counter() - start)
    elapsed = np.asarray(timings) * 1000
    return ({"p50_ms": float(np.percentile(elapsed, 50)), "p95_ms": float(np.percentile(elapsed, 95)),
             "throughput_texts_per_sec": float(len(texts) * repeats / np.sum(elapsed / 1000))},
            np.asarray(embeddings, dtype=np.float32))


def main() -> None:
    torch_model = SentenceTransformer(DEFAULT_MODEL, device="cpu")
    openvino_model = SentenceTransformer(DEFAULT_MODEL, backend="openvino", device="cpu")
    result: dict[str, object] = {
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "python": sys.version.split()[0]},
        "versions": _versions(), "model_id": DEFAULT_MODEL, "pooling": "model native Sentence Transformers pooling",
        "normalization": "L2", "warmup_runs": WARMUP, "repeats": REPEATS, "batch_sizes": {},
    }
    for batch_size in BATCH_SIZES:
        texts = SAMPLE_INPUTS[:batch_size]
        pytorch, torch_embeddings = _run(torch_model, texts, REPEATS)
        openvino, ov_embeddings = _run(openvino_model, texts, REPEATS)
        cosine = np.sum(torch_embeddings * ov_embeddings, axis=1) / np.maximum(
            np.linalg.norm(torch_embeddings, axis=1) * np.linalg.norm(ov_embeddings, axis=1), 1e-12)
        result["batch_sizes"][str(batch_size)] = {
            "texts": texts, "pytorch_cpu": pytorch, "openvino_cpu": openvino,
            "embedding_cosine_parity_mean": float(np.mean(cosine)),
            "embedding_cosine_parity_min": float(np.min(cosine)),
        }
    output = json.dumps(result, indent=2)
    print(output)
    with open("benchmark_results.json", "w", encoding="utf-8") as handle:
        handle.write(output + "\n")


if __name__ == "__main__":
    main()
