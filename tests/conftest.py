"""測試共用 fixture：建立一個暫時的專案根目錄，避免動到真正的 budget.yaml 與 data/。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def tmp_root(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "config").mkdir(parents=True)
    for name in ["settings.yaml", "budget.yaml", "boundary_main.geojson", "boundary_buffer.geojson"]:
        src = REPO / "config" / name
        if src.exists():
            shutil.copy(src, root / "config" / name)
    (root / "CLAUDE.md").write_text("test root", encoding="utf-8")
    (root / "prompts").mkdir()
    for p in (REPO / "prompts").glob("*"):
        shutil.copy(p, root / "prompts" / p.name)
    monkeypatch.setenv("DADAOCHENG_ROOT", str(root))
    from dadaocheng import config

    config._settings_cached.cache_clear()
    yield root
    config._settings_cached.cache_clear()
