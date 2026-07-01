"""Доменные модели пайплайна."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class ZoomLevel:
    level: int
    width: int
    height: int


@dataclass
class PanoramaRef:
    """Метаданные одной панорамы, достаточные для сшивки и геопривязки."""

    panoid: str
    lon: float
    lat: float
    timestamp: int
    image_id: str
    tile_size: int
    zooms: list[ZoomLevel]
    azimuth_origin: float          # Origin[0]: азимут (град) у левого края equirect
    pitch_origin: float            # Origin[1]: см. panorama.py (пока приблизительно)
    neighbor_oids: list[str] = field(default_factory=list)
    address: Optional[str] = None  # ближайший адрес из Markers

    def zoom(self, level: int) -> ZoomLevel:
        for z in self.zooms:
            if z.level == level:
                return z
        raise KeyError(f"zoom level {level} not in panorama {self.panoid}")

    @property
    def max_zoom(self) -> int:
        return min(z.level for z in self.zooms)


@dataclass
class Detection:
    """Прямоугольник баннера в нормализованных координатах панорамы [0..1]."""

    fx0: float
    fy0: float
    fx1: float
    fy1: float
    score: float = 1.0
    label: str = "banner"

    @property
    def cx(self) -> float:
        return (self.fx0 + self.fx1) / 2

    @property
    def cy(self) -> float:
        return (self.fy0 + self.fy1) / 2


@dataclass
class BannerRecord:
    """Итоговая строка в таблице/БД."""

    banner_id: str
    panoid: str
    lon: float
    lat: float
    timestamp: int
    bearing_deg: Optional[float]      # компас-азимут на баннер от точки съёмки
    category: str = "другое"
    phones: list[str] = field(default_factory=list)
    text: str = ""
    address: Optional[str] = None
    score: float = 1.0
    full_image_path: Optional[str] = None   # обзорное фото панорамы
    crop_image_path: Optional[str] = None   # вырезанный баннер (макс.зум)
    source_url: Optional[str] = None

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["phones"] = ", ".join(self.phones)
        return d
