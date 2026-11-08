"""設定載入、專案根目錄、亂數種子。

- 根目錄＝含有 CLAUDE.md 的資料夾（也可用環境變數 DADAOCHENG_ROOT 指定，測試時用）。
- 金鑰只從 .env 讀（規則 9），這裡不印出任何金鑰。
- 隨機性一律用 settings.yaml 的 seed（規則 10）。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    env = os.environ.get("DADAOCHENG_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "CLAUDE.md").exists():
            return parent
    return Path.cwd().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=4)
def _settings_cached(root: str) -> dict[str, Any]:
    return _load_yaml(Path(root) / "config" / "settings.yaml")


def settings(root: Path | None = None) -> dict[str, Any]:
    """回傳 settings.yaml 的內容（快取；修改設定後重開 Python 即可）。"""
    return _settings_cached(str(root or project_root()))


def path(key: str, root: Path | None = None) -> Path:
    """依 settings.paths 的鍵回傳絕對路徑。"""
    r = root or project_root()
    return r / settings(r)["paths"][key]


def rng(offset: int = 0) -> np.random.Generator:
    """固定種子的亂數產生器。不同用途請給不同 offset，避免彼此相關。"""
    return np.random.default_rng(int(settings()["seed"]) + offset)


def load_env(root: Path | None = None) -> None:
    """載入 .env（若存在）。不覆寫已存在的環境變數。"""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    env_file = (root or project_root()) / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def require_env(name: str) -> str:
    """取得金鑰；沒有就丟出清楚的錯誤（不印出值）。"""
    load_env()
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(
            f"環境變數 {name} 未設定。請把 .env.example 複製成 .env 並填入（見 待執行任務描述.md）。"
        )
    return val
