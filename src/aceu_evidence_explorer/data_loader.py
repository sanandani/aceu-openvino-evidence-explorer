from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .schemas import Brief, Profile


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_profiles() -> list[Profile]:
    payload = load_json(DATA_DIR / "profiles.json")
    return [Profile.from_dict(item) for item in payload.get("profiles", [])]


def load_briefs() -> list[Brief]:
    payload = load_json(DATA_DIR / "project_briefs.json")
    return [Brief.from_dict(item) for item in payload.get("briefs", [])]
