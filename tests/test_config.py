import importlib

import pytest

from aceu_evidence_explorer import config


def test_data_dir_uses_environment_override(tmp_path, monkeypatch):
    (tmp_path / "profiles.json").write_text('{"profiles": []}', encoding="utf-8")
    (tmp_path / "project_briefs.json").write_text('{"briefs": []}', encoding="utf-8")
    monkeypatch.setenv("ACEU_DATA_DIR", str(tmp_path))

    loaded = importlib.reload(config)

    assert loaded.DATA_DIR == tmp_path.resolve()


def test_missing_data_files_name_checked_path_and_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("ACEU_DATA_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError) as exc_info:
        importlib.reload(config)

    message = str(exc_info.value)
    assert str(tmp_path / "profiles.json") in message
    assert "ACEU_DATA_DIR" in message
