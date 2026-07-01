"""Загрузка конфигурации из config.yaml с переопределением через окружение."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CFG = _ROOT / "config.yaml"


class Config:
    """Тонкая обёртка над словарём конфига с точечным доступом."""

    def __init__(self, data: dict[str, Any]):
        self._d = data

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        p = Path(path) if path else _DEFAULT_CFG
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    def get(self, dotted: str, default: Any = None) -> Any:
        """cfg.get('panorama.overview_zoom')."""
        # Переопределение через окружение: BP_PANORAMA_OVERVIEW_ZOOM
        env_key = "BP_" + dotted.replace(".", "_").upper()
        if env_key in os.environ:
            return _coerce(os.environ[env_key])
        node: Any = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _coerce(v: str) -> Any:
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v
