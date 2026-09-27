from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
MAX_INPUT_CHARS = 4000
DEFAULT_RULES = {
    "strong_threshold": 0.36,
    "moderate_threshold": 0.26,
    "weak_threshold": 0.15,
}
