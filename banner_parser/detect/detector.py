"""Детекторы баннеров. Возвращают Detection в нормализованных координатах панорамы."""
from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from ..model import Detection

log = logging.getLogger(__name__)


class BannerDetector:
    """Интерфейс детектора."""

    def detect(self, overview: Image.Image) -> list[Detection]:
        raise NotImplementedError


class RegionDetector(BannerDetector):
    """Фиксированные регионы из конфига — для демо и ручной разметки.

    regions: список [fx0, fy0, fx1, fy1] в долях панорамы.
    """

    def __init__(self, regions: list[list[float]]):
        self.regions = regions or []

    def detect(self, overview: Image.Image) -> list[Detection]:
        return [Detection(*r[:4], score=1.0, label="banner") for r in self.regions]


class YoloDetector(BannerDetector):
    """YOLO по перекрывающимся перспективным видам; bbox → equirect.

    Требует ultralytics + torch и обученные веса (weights/banner_yolo.pt).
    """

    def __init__(self, weights: str, conf: float = 0.35, n_views: int = 6,
                 fov_h_deg: float = 90.0, view_size: int = 1024):
        self.weights = weights
        self.conf = conf
        self.n_views = n_views
        self.fov_h_deg = fov_h_deg
        self.view_size = view_size
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy
            self._model = YOLO(self.weights)
        return self._model

    def detect(self, overview: Image.Image) -> list[Detection]:
        from .reproject import horizon_views  # lazy (numpy)

        model = self._load()
        dets: list[Detection] = []
        for _yaw, view in horizon_views(overview, self.n_views, self.fov_h_deg,
                                        out=self.view_size):
            res = model.predict(view.image, conf=self.conf, verbose=False)[0]
            uv = view.uv_map
            for box in res.boxes:
                x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
                score = float(box.conf[0])
                det = _box_to_equirect(uv, x0, y0, x1, y1, score)
                if det is not None:
                    dets.append(det)
        return _dedup_overlaps(dets)


def _box_to_equirect(uv, x0, y0, x1, y1, score) -> Optional[Detection]:
    h, w = uv.shape[:2]
    x0, x1 = max(0, x0), min(w - 1, x1)
    y0, y1 = max(0, y0), min(h - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    fxs = uv[y0:y1, x0:x1, 0]
    fys = uv[y0:y1, x0:x1, 1]
    # регион не должен пересекать шов 0/1 по fx (иначе min/max неверны)
    if fxs.max() - fxs.min() > 0.5:
        return None
    return Detection(float(fxs.min()), float(fys.min()),
                     float(fxs.max()), float(fys.max()), score=score)


def _dedup_overlaps(dets: list[Detection], iou_thr: float = 0.4) -> list[Detection]:
    """Один щит попадает в соседние виды — склеиваем по IoU, оставляя лучший score."""
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for d in dets:
        if all(_iou(d, k) < iou_thr for k in kept):
            kept.append(d)
    return kept


def _iou(a: Detection, b: Detection) -> float:
    ix0, iy0 = max(a.fx0, b.fx0), max(a.fy0, b.fy0)
    ix1, iy1 = min(a.fx1, b.fx1), min(a.fy1, b.fy1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a.fx1 - a.fx0) * (a.fy1 - a.fy0)
    area_b = (b.fx1 - b.fx0) * (b.fy1 - b.fy0)
    return inter / (area_a + area_b - inter)


def build_detector(cfg) -> BannerDetector:
    backend = cfg.get("detector.backend", "region")
    if backend == "yolo":
        return YoloDetector(
            weights=cfg.get("detector.yolo_weights", "weights/banner_yolo.pt"),
            conf=cfg.get("detector.conf", 0.35),
        )
    return RegionDetector(cfg.get("detector.regions", []))
