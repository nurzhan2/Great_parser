"""Оркестрация: панорама → детекция → кроп → OCR → классификация → запись."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterator, Optional

from .classify import build_classifier
from .config import Config
from .detect import build_detector
from .model import BannerRecord, Detection
from .ocr import build_ocr, extract_phones
from .storage import Storage
from .verify import build_verifier
from .yandex import HttpClient, Panorama
from .yandex.graph import walk_graph
from .yandex.meta import MetaClient
from .yandex.panorama import load_panorama

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = HttpClient(cfg)
        self.meta = MetaClient(self.http)
        self.detector = build_detector(cfg)
        self.verifier = build_verifier(cfg)
        self.ocr = build_ocr(cfg)
        self.classifier = build_classifier(cfg)
        self.storage = Storage(cfg.get("storage.db_path", "data/banners.sqlite"))
        self.images_dir = Path(cfg.get("storage.images_dir", "data/images"))
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.overview_zoom = cfg.get("panorama.overview_zoom", 2)
        self.crop_zoom = cfg.get("panorama.crop_zoom", 0)
        self.workers = cfg.get("panorama.tile_workers", 16)

    # ---- одна панорама ---------------------------------------------------
    def process_panorama(self, pano: Panorama) -> list[BannerRecord]:
        # Обзор нужен только детектору (в памяти), на диск не сохраняется.
        overview = pano.stitch(self.overview_zoom)
        records: list[BannerRecord] = []
        dets = self.detector.detect(overview)
        log.info("panorama %s: %d detections", pano.ref.panoid, len(dets))
        for i, det in enumerate(dets):
            rec = self._process_detection(pano, det, i)
            if rec is not None and self.storage.save(rec):
                records.append(rec)
        return records

    def _process_detection(self, pano: Panorama, det: Detection,
                           idx: int) -> Optional[BannerRecord]:
        crop = pano.crop_detection(det, zoom=self.crop_zoom, pad=0.0)
        text = self.ocr.read(crop) if self.ocr else ""

        # Проверка: действительно ли это рекламный баннер.
        if self.verifier is not None:
            vr = self.verifier.verify(crop, text)
            if not vr.is_ad:
                log.info("отклонён (не реклама): %s [%s]", pano.ref.panoid, vr.reason)
                return None

        crop_path = self.images_dir / f"{pano.ref.panoid}_{idx}.jpg"
        crop.save(crop_path, quality=92)

        phones = extract_phones(text)
        category = self.classifier.classify(text, crop)
        bearing = pano.bearing_of(det)

        bid = hashlib.md5(
            f"{pano.ref.panoid}:{det.fx0:.4f}:{det.fy0:.4f}".encode()).hexdigest()[:12]
        return BannerRecord(
            banner_id=bid,
            panoid=pano.ref.panoid,
            lon=pano.ref.lon,
            lat=pano.ref.lat,
            timestamp=pano.ref.timestamp,
            bearing_deg=bearing,
            category=category,
            phones=phones,
            text=text,
            address=pano.ref.address,
            score=det.score,
            full_image_path=None,
            crop_image_path=str(crop_path),
            source_url=_yandex_url(pano.ref.lon, pano.ref.lat, bearing),
        )

    # ---- точка / обход ---------------------------------------------------
    def process_point(self, lon: float, lat: float) -> list[BannerRecord]:
        pano = load_panorama(self.meta, self.http, lon, lat, self.workers)
        if pano is None:
            log.warning("нет панорамы в точке %f,%f", lon, lat)
            return []
        return self.process_panorama(pano)

    def crawl(self, start_lon: float, start_lat: float) -> Iterator[BannerRecord]:
        max_nodes = self.cfg.get("crawl.max_nodes", 500)
        bbox = self.cfg.get("crawl.bbox", None)
        bbox = tuple(bbox) if bbox else None
        for ref in walk_graph(self.meta, start_lon, start_lat, max_nodes, bbox):
            pano = Panorama(ref, self.http, self.workers)
            yield from self.process_panorama(pano)

    def close(self) -> None:
        self.storage.close()


def _yandex_url(lon: float, lat: float, bearing: Optional[float]) -> str:
    d = f"{bearing:.1f}" if bearing is not None else "0"
    return (f"https://yandex.ru/maps/?l=stv,sta&panorama%5Bpoint%5D={lon},{lat}"
            f"&panorama%5Bdirection%5D={d},0&panorama%5Bfull%5D=true")
