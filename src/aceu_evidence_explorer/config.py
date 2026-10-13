from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ACEU_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
REQUIRED_DATA_FILES = ("profiles.json", "project_briefs.json")
missing_data_files = [name for name in REQUIRED_DATA_FILES if not (DATA_DIR / name).is_file()]
if not DATA_DIR.is_dir() or missing_data_files:
    checked = ", ".join(str(DATA_DIR / name) for name in missing_data_files) or str(DATA_DIR)
    raise FileNotFoundError(
        f"ACE-U data directory or required files are missing. Checked: {checked}. "
        "Set ACEU_DATA_DIR to a directory containing profiles.json and project_briefs.json."
    )
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
MAX_INPUT_CHARS = 4000
DEFAULT_RULES = {
    "strong_threshold": 0.36,
    "moderate_threshold": 0.26,
    "weak_threshold": 0.15,
}
