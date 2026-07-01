"""Работа с изображением панорамы: сшивка, кроп региона в макс.зуме, геопривязка."""
from __future__ import annotations

import concurrent.futures as cf
import io
import logging
import math
from typing import Optional

import requests
from PIL import Image

from ..model import Detection, PanoramaRef
from .meta import HttpClient

log = logging.getLogger(__name__)

TILE_URL = "https://pano.maps.yandex.net/{img}/{z}.{x}.{y}"


class Panorama:
    """Обёртка над PanoramaRef + доступ к тайлам изображения."""

    def __init__(self, ref: PanoramaRef, http: HttpClient, workers: int = 16):
        self.ref = ref
        self.http = http
        self.workers = workers

    # ---- загрузка тайлов -------------------------------------------------
    def _tile(self, z: int, x: int, y: int) -> Optional[Image.Image]:
        url = TILE_URL.format(img=self.ref.image_id, z=z, x=x, y=y)
        r = self.http.get(url)
        if r is None:
            return None
        try:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return None

    def _grid(self, jobs, place, canvas) -> int:
        ok = 0
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self._tile, *j): j for j in jobs}
            for fut in cf.as_completed(futs):
                z, x, y = futs[fut]
                tile = fut.result()
                if tile is not None:
                    canvas.paste(tile, place(x, y))
                    ok += 1
        return ok

    # ---- сшивка полной панорамы -----------------------------------------
    def stitch(self, zoom: int) -> Image.Image:
        zl = self.ref.zoom(zoom)
        ts = self.ref.tile_size
        cols, rows = math.ceil(zl.width / ts), math.ceil(zl.height / ts)
        canvas = Image.new("RGB", (zl.width, zl.height))
        jobs = [(zoom, x, y) for y in range(rows) for x in range(cols)]
        ok = self._grid(jobs, lambda x, y: (x * ts, y * ts), canvas)
        log.info("stitch %s z%d: %d/%d tiles", self.ref.panoid, zoom, ok, len(jobs))
        return canvas

    # ---- кроп региона в заданном (обычно максимальном) зуме ---------------
    def crop_region(self, fx0, fy0, fx1, fy1, zoom: Optional[int] = None,
                    pad: float = 0.0) -> Image.Image:
        """Вырезать нормализованный регион, докачивая только нужные тайлы."""
        z = self.ref.max_zoom if zoom is None else zoom
        zl = self.ref.zoom(z)
        ts = self.ref.tile_size
        fx0, fy0 = max(0.0, fx0 - pad), max(0.0, fy0 - pad)
        fx1, fy1 = min(1.0, fx1 + pad), min(1.0, fy1 + pad)
        px0, py0 = int(fx0 * zl.width), int(fy0 * zl.height)
        px1, py1 = int(fx1 * zl.width), int(fy1 * zl.height)
        tx0, ty0, tx1, ty1 = px0 // ts, py0 // ts, px1 // ts, py1 // ts
        canvas = Image.new("RGB", ((tx1 - tx0 + 1) * ts, (ty1 - ty0 + 1) * ts))
        jobs = [(z, x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
        self._grid(jobs, lambda x, y: ((x - tx0) * ts, (y - ty0) * ts), canvas)
        return canvas.crop((px0 - tx0 * ts, py0 - ty0 * ts,
                            px1 - tx0 * ts, py1 - ty0 * ts))

    def crop_detection(self, det: Detection, zoom: Optional[int] = None,
                       pad: float = 0.02) -> Image.Image:
        return self.crop_region(det.fx0, det.fy0, det.fx1, det.fy1, zoom=zoom, pad=pad)

    # ---- геопривязка -----------------------------------------------------
    def fx_to_azimuth(self, fx: float) -> float:
        """Горизонтальная доля [0..1] → компас-азимут (град).

        Калибровано на реальной панораме: fx = ((az - Origin[0]) mod 360)/360.
        """
        return (self.ref.azimuth_origin + fx * 360.0) % 360.0

    def azimuth_to_fx(self, az: float) -> float:
        return ((az - self.ref.azimuth_origin) % 360.0) / 360.0

    def bearing_of(self, det: Detection) -> float:
        """Компас-азимут (град) на центр баннера от точки съёмки."""
        return self.fx_to_azimuth(det.cx)


def load_panorama(meta_client, http: HttpClient, lon: float, lat: float,
                  workers: int = 16) -> Optional[Panorama]:
    raw = meta_client.by_coords(lon, lat)
    if raw is None:
        return None
    ref = meta_client.parse(raw)
    return Panorama(ref, http, workers=workers)
