"""Оркестрация: панорама → детекция → кроп → OCR → классификация → запись."""
from __future__ import annotations

import hashlib
import logging
import math
import time
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

from . import runlog
from .classify import build_classifier
from .config import Config
from .detect import build_detector
from .model import BannerRecord, Detection
from .ocr import build_ocr, extract_phones
from .storage import Storage
from .verify import build_verifier
from .yandex import HttpClient, Panorama
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
        self.storage = Storage(
            cfg.get("storage.db_path", "data/banners.sqlite"),
            dedup_radius_m=cfg.get("dedup.radius_m", 25.0),
            dedup_phone_radius_m=cfg.get("dedup.phone_radius_m", 300.0),
            dedup_assumed_distance_m=cfg.get("dedup.assumed_distance_m", 25.0))
        self.images_dir = Path(cfg.get("storage.images_dir", "data/images"))
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.overview_zoom = cfg.get("panorama.overview_zoom", 2)
        self.crop_zoom = cfg.get("panorama.crop_zoom", 0)
        self.workers = cfg.get("panorama.tile_workers", 16)
        self.max_per_panorama = cfg.get("detector.max_per_panorama", 10)
        # Кандидатов на разбор больше, чем сохраняемых: часть отсеет verify.
        self.max_candidates = cfg.get("detector.max_candidates", 30)
        # Отбраковка по размеру будущего кропа. Размер НЕ разделяет рекламу
        # и мусор (замерено: медиана минимальной стороны 143 px у настоящих
        # против 184 px у ложных), поэтому пороги ставим только там, где кроп
        # заведомо непригоден: слишком мелкий — нечитаем, слишком крупный —
        # это забор или стена во весь кадр, а не щит.
        self.min_crop_side = cfg.get("detector.min_crop_side", 50)
        self.min_crop_area = cfg.get("detector.min_crop_area", 5000)
        self.max_crop_area = cfg.get("detector.max_crop_area", 800_000)
        self.crop_pad = cfg.get("detector.crop_pad", 0.0)

    # ---- одна панорама ---------------------------------------------------
    def process_panorama(self, pano: Panorama) -> list[BannerRecord]:
        pid = pano.ref.panoid
        # Обзор нужен только детектору (в памяти), на диск не сохраняется.
        runlog.set_stage(f"сшивка {pid[:16]}")
        t0 = time.monotonic()
        overview = pano.stitch(self.overview_zoom)
        t_stitch = time.monotonic() - t0

        # Детекция — самый тяжёлый этап и по времени, и по памяти: если процесс
        # убивают по OOM, это происходит здесь, поэтому этап отмечен явно.
        runlog.set_stage(f"детекция {pid[:16]}")
        t0 = time.monotonic()
        # Разбираем по убыванию score — чтобы под потолок попали лучшие.
        dets = sorted(self.detector.detect(overview), key=lambda d: d.score, reverse=True)
        t_detect = time.monotonic() - t0
        log.info("panorama %s: %d detections (сшивка %.1f с, детекция %.1f с, rss %s)",
                 pid, len(dets), t_stitch, t_detect, runlog.rss_str())

        # Раньше здесь стоял return после первой сохранённой детекции, и
        # ограждение с десятком рекламных секций давало одну запись за визит.
        # Теперь берём все прошедшие, но с двумя потолками: сколько кандидатов
        # вообще разбирать (кроп+OCR — дорогие) и сколько записей сохранить.
        saved: list[BannerRecord] = []
        for i, det in enumerate(dets[:self.max_candidates]):
            if len(saved) >= self.max_per_panorama:
                log.info("панорама %s: достигнут потолок %d баннеров, "
                         "остальные %d детекций не разбираем",
                         pid, self.max_per_panorama, len(dets) - i)
                break
            rec, crop = self._process_detection(pano, det, i)
            if rec is None or not self.storage.save(rec):
                continue
            # Файл пишем ТОЛЬКО после того, как запись принята: иначе на диск
            # ложится кроп каждого кандидата, включая отсеянных дедупом —
            # на тестовой панораме это 24 файла при 5 записях.
            crop.save(rec.crop_image_path, quality=92)
            saved.append(rec)
        if len(dets) > self.max_candidates:
            log.info("панорама %s: разобрано %d кандидатов из %d (потолок разбора)",
                     pid, self.max_candidates, len(dets))
        log.info("панорама %s: сохранено %d баннеров", pid, len(saved))
        return saved

    def _process_detection(self, pano: Panorama, det: Detection,
                           idx: int) -> tuple[Optional[BannerRecord], Optional[Image.Image]]:
        """Возвращает (запись, кроп). Файл не пишет — это делает вызывающий
        после того, как запись прошла дедуп."""
        runlog.set_stage(f"кроп {pano.ref.panoid[:16]}")
        # Размер считаем до скачивания тайлов — непригодные отсеиваем бесплатно.
        w, h = pano.crop_size_px(det, zoom=self.crop_zoom, pad=self.crop_pad)
        if min(w, h) < self.min_crop_side or w * h < self.min_crop_area:
            log.info("отброшен (кроп мелкий %dx%d): %s", w, h, pano.ref.panoid)
            return None, None
        if self.max_crop_area and w * h > self.max_crop_area:
            log.info("отброшен (кроп огромный %dx%d — похоже на забор/стену "
                     "во весь кадр, а не щит): %s", w, h, pano.ref.panoid)
            return None, None
        crop = pano.crop_detection(det, zoom=self.crop_zoom, pad=self.crop_pad)

        # Сначала дешёвая проверка «это реклама» — до OCR.
        if self.verifier is not None:
            vr = self.verifier.verify(crop)
            if not vr.is_ad:
                log.info("отклонён (не реклама): %s [%s]", pano.ref.panoid, vr.reason)
                return None, None

        # OCR только для прошедших проверку кропов.
        runlog.set_stage(f"OCR {pano.ref.panoid[:16]}")
        text = self.ocr.read(crop) if self.ocr else ""
        category = self.classifier.classify(text, crop)
        # Фильтра по теме здесь БОЛЬШЕ НЕТ. «Это реклама» решает детекция и
        # verify, «это недвижимость» — тема по тексту; смешивать их нельзя:
        # баннер с нечитаемым OCR терял тему и выбрасывался (5 из 8 кандидатов
        # на замеренной панораме). Собранное всегда можно отфильтровать при
        # выгрузке, несобранное уже не вернуть — фильтр переехал в export.

        crop_path = self.images_dir / f"{pano.ref.panoid}_{idx}.jpg"
        phones = extract_phones(text)
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
        ), crop

    # ---- точка / обход ---------------------------------------------------
    def process_point(self, lon: float, lat: float) -> list[BannerRecord]:
        runlog.set_stage(f"meta-запрос точки {lon:.4f},{lat:.4f}")
        pano = load_panorama(self.meta, self.http, lon, lat, self.workers)
        if pano is None:
            log.warning("нет панорамы в точке %f,%f — meta-API вернул пусто "
                        "(либо там действительно нет съёмки, либо запрос отклонён; "
                        "выше в логе должна быть причина от HTTP-слоя)", lon, lat)
            return []
        return self.process_panorama(pano)

    def crawl(self, start_lon: float, start_lat: float,
              max_panoramas: Optional[int] = None) -> Iterator[BannerRecord]:
        """Резюмируемый обход графа панорам в пределах bbox (напр. Москва).

        Состояние (посещённые + очередь) хранится в БД — после остановки/сбоя
        обход продолжается с того же места, без повторного скачивания.
        max_panoramas — необязательный лимит на число панорам за один запуск
        (None = без лимита, крутить пока не исчерпается очередь).
        """
        bbox = self.cfg.get("crawl.bbox", None)
        bbox = tuple(bbox) if bbox else None
        min_dist = self.cfg.get("crawl.min_distance_m", 250)
        st = self.storage

        # Состояние ДО старта: сразу видно, продолжаем мы обход или сеем заново,
        # и не пуста ли очередь (пустая очередь = обход уже исчерпан, а не сломан).
        log.info("обход: bbox=%s, min_distance=%s м, лимит за запуск=%s",
                 bbox or "без рамки", min_dist, max_panoramas or "нет")
        log.info("состояние до старта: посещено %d, в очереди %d, баннеров в БД %d",
                 st.visited_count(), st.frontier_pending(), st.count())

        # Посев стартовой точки. Раньше он срабатывал только на пустом
        # состоянии, и запуск с новыми --lon/--lat молча игнорировал
        # координаты, продолжая старый обход. Теперь сеем всегда, если эта
        # панорама ещё не посещалась — так в общую БД можно добавить второй
        # и третий старт. Цена: один лишний meta-запрос на запуск.
        runlog.set_stage("посев стартовой точки")
        raw = self.meta.by_coords(start_lon, start_lat)
        if raw is None:
            log.warning("стартовая точка %f,%f: meta не отдала панораму — "
                        "посев пропущен, идём по существующей очереди",
                        start_lon, start_lat)
        else:
            ref = self.meta.parse(raw)
            if st.is_visited(ref.panoid):
                log.info("стартовая панорама %s уже посещалась — "
                         "продолжаем существующий обход", ref.panoid)
            elif not _in_bbox(ref.lon, ref.lat, bbox):
                log.warning("стартовая точка %f,%f вне bbox %s — посев пропущен",
                            ref.lon, ref.lat, bbox)
            else:
                log.info("посев новой стартовой точки %f,%f (панорама %s)",
                         ref.lon, ref.lat, ref.panoid)
                st.mark_visited(ref.panoid)
                st.enqueue(ref.neighbor_oids)
                clat, clon = _cell_coords(ref.lon, ref.lat, min_dist)
                st.mark_cell(f"{clat}:{clon}", ref.lon, ref.lat)
                yield from self.process_panorama(Panorama(ref, self.http, self.workers))
            st.commit()

        processed = 0
        meta_fails = 0          # подряд идущие отказы meta-API — признак блокировки
        t_start = time.monotonic()
        while True:
            runlog.set_stage("выбор следующей панорамы")
            oid = st.next_oid()
            if oid is None:
                log.info("frontier пуст — обход завершён (%d панорам, %d баннеров). "
                         "Это нормальное окончание: граф в пределах bbox исчерпан. "
                         "Чтобы продолжить — задайте другую стартовую точку или bbox",
                         st.visited_count(), st.count())
                break
            runlog.set_stage(f"meta-запрос {oid[:16]}")
            raw = self.meta.by_oid(oid)
            st.mark_oid_done(oid)
            if raw is None:
                meta_fails += 1
                # 30 отказов подряд — это уже не «нет данных по точке», а
                # блокировка/сеть. Молча крутиться в этом цикле бессмысленно.
                if meta_fails >= 30:
                    log.error("meta-API не отвечает %d раз подряд — обход остановлен. "
                              "Похоже на блокировку по IP или отсутствие сети; "
                              "проверьте доступность %s и настройте http.proxies",
                              meta_fails, "api-maps.yandex.ru")
                    st.commit()
                    break
                if meta_fails % 10 == 0:
                    log.warning("meta-API: %d отказов подряд", meta_fails)
                st.commit()
                continue
            meta_fails = 0
            ref = self.meta.parse(raw)
            if st.is_visited(ref.panoid):
                st.commit()
                continue
            st.mark_visited(ref.panoid)
            if not _in_bbox(ref.lon, ref.lat, bbox):
                st.commit()          # за пределами рамки — не разворачиваем дальше
                continue
            # Соседей ставим в очередь всегда — чтобы продолжать движение.
            st.enqueue(ref.neighbor_oids)
            # Детекцию запускаем только если рядом (< min_dist) ещё не снимали.
            clat, clon = _cell_coords(ref.lon, ref.lat, min_dist)
            near = st.points_in_cells(_neighbor_keys(clat, clon))
            if _too_close(ref.lon, ref.lat, near, min_dist):
                st.commit()
                continue
            st.mark_cell(f"{clat}:{clon}", ref.lon, ref.lat)
            try:
                yield from self.process_panorama(Panorama(ref, self.http, self.workers))
            except Exception:        # noqa: BLE001 — обход не должен падать на одной панораме
                # С трассировкой: раньше здесь оставалась одна строка без стека,
                # и по логу нельзя было понять, что именно сломалось.
                log.exception("ошибка обработки панорамы %s (%.5f,%.5f) — пропускаем",
                              ref.panoid, ref.lon, ref.lat)
            st.commit()
            processed += 1
            if processed % 20 == 0:
                rate = processed / max(1e-9, (time.monotonic() - t_start) / 60)
                log.info("обход: посещено %d, снято точек %d (%.1f точек/мин), "
                         "в очереди %d, баннеров %d, rss %s",
                         st.visited_count(), processed, rate,
                         st.frontier_pending(), st.count(), runlog.rss_str())
            if max_panoramas and processed >= max_panoramas:
                log.info("достигнут лимит %d обработанных точек за запуск", max_panoramas)
                break

    def close(self) -> None:
        self.storage.close()


def _in_bbox(lon: float, lat: float, bbox) -> bool:
    if not bbox:
        return True
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def _cell_coords(lon: float, lat: float, dist_m: float) -> tuple[int, int]:
    """Индексы ячейки сетки со стороной ~dist_m метров."""
    dlat = dist_m / 111_320.0
    dlon = dist_m / (111_320.0 * max(0.1, math.cos(math.radians(lat))))
    return round(lat / dlat), round(lon / dlon)


def _neighbor_keys(clat: int, clon: int) -> list[str]:
    """Ключи ячейки и её 8 соседей (окрестность 3×3)."""
    return [f"{clat + i}:{clon + j}" for i in (-1, 0, 1) for j in (-1, 0, 1)]


def _haversine_m(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    mlat = math.radians((lat0 + lat1) / 2)
    dx = (lon1 - lon0) * 111_320.0 * math.cos(mlat)
    dy = (lat1 - lat0) * 111_320.0
    return math.hypot(dx, dy)


def _too_close(lon: float, lat: float, points, min_dist: float) -> bool:
    return any(_haversine_m(lon, lat, plon, plat) < min_dist for plon, plat in points)


def _yandex_url(lon: float, lat: float, bearing: Optional[float]) -> str:
    d = f"{bearing:.1f}" if bearing is not None else "0"
    return (f"https://yandex.ru/maps/?l=stv,sta&panorama%5Bpoint%5D={lon},{lat}"
            f"&panorama%5Bdirection%5D={d},0&panorama%5Bfull%5D=true")
