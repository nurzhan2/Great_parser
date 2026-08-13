"""Клиент Яндекс.Панорам: meta-API и HTTP с ретраями/анти-блоком.

ВНИМАНИЕ: массовая выгрузка панорам нарушает пользовательское соглашение
Яндекса. Используйте с пониманием рисков (блокировки IP, юридические претензии).
Параметры min_delay / proxies в config.yaml — мера снижения нагрузки.
"""
from __future__ import annotations

import itertools
import logging
import time
from typing import Any, Optional

import requests

from ..model import PanoramaRef, ZoomLevel

log = logging.getLogger(__name__)

META_URL = "https://api-maps.yandex.ru/services/panoramas/1.x/"


class HttpClient:
    """Сессия с ретраями, паузой и ротацией прокси."""

    def __init__(self, cfg):
        self.timeout = cfg.get("http.timeout", 25)
        self.max_retries = cfg.get("http.max_retries", 3)
        self.min_delay = cfg.get("http.min_delay", 0.15)
        self._headers = {
            "User-Agent": cfg.get("http.user_agent", "Mozilla/5.0"),
            "Referer": cfg.get("http.referer", "https://yandex.ru/maps/"),
        }
        proxies = cfg.get("http.proxies", []) or []
        self._proxy_cycle = itertools.cycle(proxies) if proxies else None
        self._session = requests.Session()
        # пул под многопоточную загрузку тайлов (иначе warning'и и таймауты)
        pool = max(16, cfg.get("panorama.tile_workers", 16)) + 4
        adapter = requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._last = 0.0

    def _throttle(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.min_delay:
            time.sleep(self.min_delay - dt)
        self._last = time.monotonic()

    def _proxies(self) -> Optional[dict]:
        if not self._proxy_cycle:
            return None
        p = next(self._proxy_cycle)
        return {"http": p, "https": p}

    def get(self, url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
        last = "неизвестно"
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                r = self._session.get(
                    url, params=params, headers=self._headers,
                    proxies=self._proxies(), timeout=self.timeout,
                )
                if r.status_code == 200:
                    return r
                last = f"HTTP {r.status_code}"
                # 429/403 — это не «сбой сети», а отказ Яндекса обслуживать нас.
                # Раньше это тонуло в общем warning и выглядело как случайность.
                if r.status_code in (403, 429):
                    log.warning("HTTP %s для %s (попытка %d) — похоже на блокировку: "
                                "увеличьте http.min_delay или задайте http.proxies",
                                r.status_code, url.split("?")[0], attempt + 1)
                else:
                    log.warning("HTTP %s для %s (попытка %d)",
                                r.status_code, url.split("?")[0], attempt + 1)
            except requests.RequestException as e:
                last = type(e).__name__
                log.warning("ошибка запроса %s: %s (попытка %d)",
                            type(e).__name__, e, attempt + 1)
            time.sleep(0.5 * (attempt + 1))
        # Раньше здесь молча возвращался None, и выше по стеку это выглядело
        # как «нет панорамы» — причина отказа в лог не попадала вообще.
        log.error("все %d попытки исчерпаны (%s): %s",
                  self.max_retries, last, url.split("?")[0])
        return None


class MetaClient:
    """Запрос и разбор метаданных панорам."""

    def __init__(self, http: HttpClient):
        self.http = http

    def _fetch(self, params: dict) -> Optional[dict]:
        base = {"l": "stv", "lang": "ru_RU"}
        base.update(params)
        r = self.http.get(META_URL, params=base)
        if r is None:
            return None                     # причина уже записана в HttpClient.get
        try:
            j = r.json()
        except ValueError:
            log.warning("meta-API вернул не-JSON (%d байт, начало: %.80r)",
                        len(r.content), r.text[:80])
            return None
        if j.get("status") != "success":
            log.debug("meta-API status=%s для %s", j.get("status"), params)
            return None
        return j.get("data")

    def by_coords(self, lon: float, lat: float) -> Optional[dict]:
        return self._fetch({
            "ll": f"{lon},{lat}",
            "origin": "userAction",
            "provider": "searchByCoords",
        })

    def by_oid(self, oid: str) -> Optional[dict]:
        return self._fetch({"oid": oid, "provider": "streetview"})

    @staticmethod
    def parse(raw: dict) -> PanoramaRef:
        D = raw["Data"]
        images = D["Images"]
        zooms = [ZoomLevel(z["level"], z["width"], z["height"]) for z in images["Zooms"]]
        coords = D["Point"]["coordinates"]
        origin = D.get("EquirectangularProjection", {}).get("Origin", [0.0, 0.0])

        ann = raw.get("Annotation", {})
        oids = _extract_neighbor_oids(ann)
        address = _nearest_address(ann, coords)

        return PanoramaRef(
            panoid=D["panoramaId"],
            lon=float(coords[0]),
            lat=float(coords[1]),
            timestamp=int(D.get("timestamp", 0)),
            image_id=images["imageId"],
            tile_size=images["Tiles"]["width"],
            zooms=zooms,
            azimuth_origin=float(origin[0]),
            pitch_origin=float(origin[1]),
            neighbor_oids=oids,
            address=address,
        )


def _extract_neighbor_oids(ann: dict) -> list[str]:
    oids: list[str] = []
    for th in ann.get("Thoroughfares", []) + ann.get("Connections", []):
        href = th.get("Connection", {}).get("href", "")
        oid = _oid_from_href(href)
        if oid:
            oids.append(oid)
    return oids


def _oid_from_href(href: str) -> Optional[str]:
    # ...panoramas/1.x?oid=<OID>&provider=streetview&...
    marker = "oid="
    i = href.find(marker)
    if i == -1:
        return None
    tail = href[i + len(marker):]
    return tail.split("&", 1)[0] or None


def _nearest_address(ann: dict, coords: list[float]) -> Optional[str]:
    best: Optional[str] = None
    best_d = float("inf")
    lon0, lat0 = coords[0], coords[1]
    for m in ann.get("Markers", []):
        desc = m.get("properties", {}).get("description")
        g = m.get("geometry", {}).get("coordinates")
        if not desc or not g:
            continue
        d = (g[0] - lon0) ** 2 + (g[1] - lat0) ** 2
        if d < best_d:
            best_d, best = d, desc
    return best
