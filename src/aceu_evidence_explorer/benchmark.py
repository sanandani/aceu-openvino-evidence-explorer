from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SAMPLE_INPUTS = [
    "We need a public-interest systems engineer with mentorship and reliability experience.",
    "We need a human-centered AI review lead focused on evaluation design and uncertainty communication.",
    "We need someone who can troubleshoot edge deployments and support operational teams.",
]


def _get_env() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
    }


def _run_benchmark(model_backend: str, repeats: int = 30, warmup: int = 5) -> dict[str, float | int | str | dict[str, str]]:
    if model_backend == "pytorch":
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    else:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="openvino", device="CPU")

    timings = []
    for _ in range(warmup):
        _ = model.encode(SAMPLE_INPUTS, normalize_embeddings=True, convert_to_numpy=True)

    for _ in range(repeats):
        start = time.perf_counter()
        _ = model.encode(SAMPLE_INPUTS, normalize_embeddings=True, convert_to_numpy=True)
        timings.append(time.perf_counter() - start)

    arr = np.asarray(timings, dtype=np.float64)
    throughput = (len(SAMPLE_INPUTS) * repeats) / float(np.sum(arr))
    return {
        "backend": model_backend,
        "warmup": warmup,
        "repeats": repeats,
        "p50_ms": float(np.percentile(arr * 1000, 50)),
        "p95_ms": float(np.percentile(arr * 1000, 95)),
        "throughput_per_sec": float(throughput),
        "avg_ms": float(np.mean(arr * 1000)),
        "env": _get_env(),
    }


def main() -> None:
    results = {
        "benchmark": {
            "inputs": SAMPLE_INPUTS,
            "batch_size": len(SAMPLE_INPUTS),
            "results": [
                _run_benchmark("pytorch"),
                _run_benchmark("openvino"),
            ],
        }
    }
    print(json.dumps(results, indent=2))
    with open("benchmark_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
