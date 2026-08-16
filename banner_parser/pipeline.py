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
from .brands import build_brand_directory
from .classify import build_classifier
from .config import Config
from .detect import build_detector
from .model import BannerRecord, Detection
from .ocr import OcrResult, build_ocr, extract_contacts
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
        self.brands = build_brand_directory(cfg)
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
        # Отступ вокруг бокса: 0 обрезает текст по краям (видно на 77b7323c281d).
        self.crop_pad = cfg.get("detector.crop_pad", 0.03)
        # Выпрямление кропа в перспективу: бокс приходит из перспективного вида,
        # а режется по прямоугольнику в equirect — отсюда трапеции и раздувание.
        self.rectify = cfg.get("detector.rectify", True)
        # Ниже этой стороны кропа контакты считаются ненадёжными.
        self.contacts_min_side = cfg.get("ocr.contacts_min_side", 220)
        # Не тратим облачный OCR на кропы, где контакты физически
        # недостаточно крупные для надёжного чтения.
        self.cloud_ocr_min_side = cfg.get(
            "ocr.cloud_min_side",
            self.contacts_min_side,
        )
        # Жёсткий бюджетный предохранитель на одну панораму.
        self.max_cloud_ocr_per_panorama = cfg.get(
            "ocr.max_cloud_calls_per_panorama",
            4,
        )
        self._cloud_ocr_calls = 0

    # ---- одна панорама ---------------------------------------------------
    def process_panorama(self, pano: Panorama) -> list[BannerRecord]:
        pid = pano.ref.panoid
        # Счётчик платных OCR-вызовов отдельный для каждой панорамы.
        self._cloud_ocr_calls = 0
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
        w, h = pano.crop_size_px(det, zoom=self.crop_zoom, pad=0.0)
        if min(w, h) < self.min_crop_side or w * h < self.min_crop_area:
            log.info("отброшен (кроп мелкий %dx%d): %s", w, h, pano.ref.panoid)
            return None, None
        if self.max_crop_area and w * h > self.max_crop_area:
            log.info("отброшен (кроп огромный %dx%d — похоже на забор/стену "
                     "во весь кадр, а не щит): %s", w, h, pano.ref.panoid)
            return None, None
        # Отступ считаем от размера БОКСА, а не панорамы: доля панорамы здесь
        # бессмысленна — 3% от 17664 px это +530 px с каждой стороны.
        px, py = (det.fx1 - det.fx0) * self.crop_pad, (det.fy1 - det.fy0) * self.crop_pad
        bx0, by0 = max(0.0, det.fx0 - px), max(0.0, det.fy0 - py)
        bx1, by1 = min(1.0, det.fx1 + px), min(1.0, det.fy1 + py)
        crop = pano.crop_region(bx0, by0, bx1, by1, zoom=self.crop_zoom, pad=0.0)
        if self.rectify:
            from .detect.reproject import rectify_region
            zl = pano.ref.zoom(self.crop_zoom)
            crop = rectify_region(crop, (bx0, by0, bx1, by1),
                                  (det.fx0, det.fy0, det.fx1, det.fy1),
                                  zl.width, zl.height)

        # Сначала дешёвая проверка «это реклама» — до OCR.
        if self.verifier is not None:
            vr = self.verifier.verify(crop)
            if not vr.is_ad:
                log.info("отклонён (не реклама): %s [%s]", pano.ref.panoid, vr.reason)
                return None, None

        # OCR только для прошедших проверку кропов.
        runlog.set_stage(f"OCR {pano.ref.panoid[:16]}")
        ocr_res = None

        if self.ocr is not None:
            backend_name = getattr(
                getattr(self.ocr, "backend", None),
                "name",
                "",
            )
        
            is_cloud_ocr = (
                backend_name.startswith("vlm-")
                or backend_name.startswith("openai-")
            )
        
            # На маленьком кропе платный VLM не вызываем вообще:
            # надёжный телефон из нескольких пикселей всё равно не получить.
            if is_cloud_ocr and min(crop.size) < self.cloud_ocr_min_side:
                log.info(
                    "облачный OCR пропущен: мелкий кроп %dx%d < %d: %s",
                    crop.size[0],
                    crop.size[1],
                    self.cloud_ocr_min_side,
                    pano.ref.panoid,
                )
        
                ocr_res = OcrResult(
                    engine="skipped-lowres",
                )
        
            # Даже если verifier пропустил много кандидатов,
            # стоимость одной панорамы имеет жёсткий потолок.
            elif (
                is_cloud_ocr
                and self._cloud_ocr_calls >= self.max_cloud_ocr_per_panorama
            ):
                log.info(
                    "облачный OCR пропущен: лимит %d вызовов на панораму: %s",
                    self.max_cloud_ocr_per_panorama,
                    pano.ref.panoid,
                )
        
                ocr_res = OcrResult(
                    engine="skipped-budget",
                )
        
            else:
                if is_cloud_ocr:
                    self._cloud_ocr_calls += 1
        
                ocr_res = self.ocr.recognize(crop)
                
        text = ocr_res.text if ocr_res else ""
        if ocr_res is not None and ocr_res.failed:
            log.info("текст не распознан (движок %s отказал): %s",
                     ocr_res.engine, pano.ref.panoid)
        # Тема от движка приоритетнее словаря: VLM видит картинку, а словарь
        # работает по тексту, которого может не быть вовсе. Словарь — запасной.
        category = (ocr_res.category if ocr_res and ocr_res.category
                    else self.classifier.classify(text, crop))
        advertiser = ocr_res.advertiser if ocr_res else None
        construction = ocr_res.construction if ocr_res else None
        # Нормализация бренда и обогащение контактов из справочника.
        # Контакты справочника НЕ смешиваются с прочитанными со щита:
        # это данные другой природы и другой надёжности.
        # ЖК со щита — подсказка для поиска застройщика: «Квартал Домашний»
        # однозначно указывает на Самолёт, даже если сам застройщик не назван.
        binfo = (self.brands.lookup(advertiser, ocr_res.complex_name if ocr_res else None)
                 if self.brands else None)
        # Частное объявление в общую базу контактов не идёт: это физлицо,
        # а не рекламодатель рынка. Считаем таким, если движок так сказал.
        personal = bool(ocr_res and (ocr_res.advertiser_type or "").lower().startswith("частн"))
        if binfo is not None and binfo.matched and not category:
            category = binfo.category or category
        # Фильтра по теме здесь БОЛЬШЕ НЕТ. «Это реклама» решает детекция и
        # verify, «это недвижимость» — тема по тексту; смешивать их нельзя:
        # баннер с нечитаемым OCR терял тему и выбрасывался (5 из 8 кандидатов
        # на замеренной панораме). Собранное всегда можно отфильтровать при
        # выгрузке, несобранное уже не вернуть — фильтр переехал в export.

        crop_path = self.images_dir / f"{pano.ref.panoid}_{idx}.jpg"
        contacts = extract_contacts(text)
        # Мелкий шрифт на далёком щите физически нечитаем: на кропе 291x108 px
        # телефон имеет высоту в несколько пикселей. Замер показал, что на таких
        # кропах распознавание даёт ПРАВДОПОДОБНЫЕ, но неверные контакты —
        # телефон МВД 8(985)277-78-79 превратился в 227-78-70, kinomax.ru в
        # kinomaks.ru. Звонить по такому номеру попадёшь не туда, поэтому
        # контакты с кропов ниже порога помечаются ненадёжными, а не выдаются
        # как проверенные.
        if min(crop.size) < self.contacts_min_side:
            if contacts.phones or contacts.sites or contacts.telegram:
                log.info("контакты с мелкого кропа %dx%d помечены ненадёжными: %s",
                         crop.size[0], crop.size[1], pano.ref.panoid)
            contacts.phones_unreliable = contacts.phones_unreliable + contacts.phones
            contacts.phones = []
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
            phones=contacts.phones,
            phones_unreliable=contacts.phones_unreliable,
            sites=contacts.sites,
            telegram=contacts.telegram,
            text=text,
            advertiser=advertiser,
            brand=(binfo.canonical if binfo else None),
            brand_matched=bool(binfo and binfo.matched),
            construction=construction,
            dir_site=(binfo.site if binfo else None),
            dir_phone=(binfo.phone if binfo else None),
            is_realty=(ocr_res.is_realty if ocr_res else None),
            developer=((binfo.canonical if binfo and binfo.matched else None)
                       or (ocr_res.developer if ocr_res else None)),
            complex_name=((ocr_res.complex_name if ocr_res else None)
                          or (binfo.complex_name if binfo else None)),
            offer_type=(ocr_res.offer_type if ocr_res else None),
            advertiser_type=(ocr_res.advertiser_type if ocr_res else None),
            personal_ad=personal,
            ocr_engine=(ocr_res.engine if ocr_res else None),
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
        if self.ocr is not None:
            log.info("OCR %s", self.ocr.stats())
            self.ocr.close()
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
