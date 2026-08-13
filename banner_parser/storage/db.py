"""SQLite-хранилище баннеров с дедупликацией одного щита из разных панорам."""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Iterator

from ..model import BannerRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS banners (
    banner_id   TEXT PRIMARY KEY,
    dedup_key   TEXT,
    panoid      TEXT,
    lon         REAL,
    lat         REAL,
    timestamp   INTEGER,
    bearing_deg REAL,
    category    TEXT,
    phones      TEXT,
    text        TEXT,
    address     TEXT,
    score       REAL,
    full_image_path TEXT,
    crop_image_path TEXT,
    source_url  TEXT
);
CREATE INDEX IF NOT EXISTS idx_dedup ON banners(dedup_key);
CREATE INDEX IF NOT EXISTS idx_category ON banners(category);
-- Дедуп ищет соседей по координатам — без индекса это скан всей таблицы.
CREATE INDEX IF NOT EXISTS idx_geo ON banners(lat, lon);

-- Состояние обхода для резюмируемости (переживает остановку/сбой).
CREATE TABLE IF NOT EXISTS crawl_frontier (
    oid  TEXT PRIMARY KEY,
    done INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_frontier_done ON crawl_frontier(done);
CREATE TABLE IF NOT EXISTS crawl_visited (
    panoid TEXT PRIMARY KEY
);
-- Снятые точки по ячейкам сетки: не снимаем баннер ближе min_distance_m
-- (координаты храним для точной проверки расстояния по окрестности).
CREATE TABLE IF NOT EXISTS crawl_cells (
    cell TEXT PRIMARY KEY,
    lon  REAL,
    lat  REAL
);
"""


def dedup_key(rec: BannerRecord) -> str:
    """Грубый ключ — только для человека в таблице. Решение о дубликате
    принимает Storage.find_duplicate(): ключ по округлению склеивал разные
    щиты и не склеивал один и тот же с соседних панорам."""
    b = round((rec.bearing_deg or 0) / 10) * 10
    return f"geo:{round(rec.lon, 4)}:{round(rec.lat, 4)}:{b}:{rec.category}"


def _haversine_m(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    mlat = math.radians((lat0 + lat1) / 2)
    dx = (lon1 - lon0) * 111_320.0 * math.cos(mlat)
    dy = (lat1 - lat0) * 111_320.0
    return math.hypot(dx, dy)


def estimated_position(lon: float, lat: float, bearing_deg: float | None,
                       assumed_distance_m: float) -> tuple[float, float]:
    """Грубая оценка положения САМОГО щита: точка съёмки плюс луч по азимуту.

    Дальность до щита из панорамы не измеряется, поэтому берём типичную —
    щиты и ограждения стоят у дороги в нескольких десятках метров. Без этого
    дедуп невозможен в принципе: у всех детекций с одной панорамы координаты
    камеры одинаковы, и кластеризация по ним склеила бы разные щиты, видимые
    с одной точки, в один объект.
    """
    if bearing_deg is None:
        return lon, lat
    a = math.radians(bearing_deg)
    dx = assumed_distance_m * math.sin(a)          # восток
    dy = assumed_distance_m * math.cos(a)          # север
    dlat = dy / 111_320.0
    dlon = dx / (111_320.0 * max(0.1, math.cos(math.radians(lat))))
    return lon + dlon, lat + dlat


class Storage:
    def __init__(self, db_path: str, dedup_radius_m: float = 25.0,
                 dedup_phone_radius_m: float = 300.0,
                 dedup_assumed_distance_m: float = 25.0):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self.dedup_radius_m = dedup_radius_m
        self.dedup_phone_radius_m = dedup_phone_radius_m
        self.dedup_assumed_distance_m = dedup_assumed_distance_m

    # ---- дедупликация ----------------------------------------------------
    def find_duplicate(self, rec: BannerRecord):
        """Ищет уже сохранённый тот же физический щит. Возвращает строку или None.

        Три болезни лечатся здесь:
          • один щит с соседних панорам не склеивался — старый ключ округлял
            координаты до ~11 м и азимут до 10°, и съёмка того же щита с точки
            в 20 метрах давала другой ключ;
          • разные щиты с одним телефоном схлопывались — телефон был
            самостоятельным ключом, и федеральная кампания с единым номером
            убила бы базу по стране. Теперь телефон работает только вместе
            с географией, зато с бо́льшим радиусом: один щит видно и издалека;
          • кластеризация по координатам КАМЕРЫ склеивала разные щиты, видимые
            с одной панорамы: у них координаты совпадают точно. Поэтому
            сравниваем оценку положения самого щита (камера + луч по азимуту).

        Тема в сравнении не участвует: она приходит из OCR, часто не определена
        и меняется от кадра к кадру — привязывать к ней тождество объекта нельзя.
        """
        px, py = estimated_position(rec.lon, rec.lat, rec.bearing_deg,
                                    self.dedup_assumed_distance_m)
        # Прямоугольник поиска берём с запасом: камера может стоять дальше,
        # чем оценка щита, поэтому прибавляем предполагаемую дальность.
        r = max(self.dedup_radius_m, self.dedup_phone_radius_m) + self.dedup_assumed_distance_m
        dlat = r / 111_320.0
        dlon = r / (111_320.0 * max(0.1, math.cos(math.radians(rec.lat))))
        rows = self.conn.execute(
            "SELECT * FROM banners WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (rec.lat - dlat, rec.lat + dlat, rec.lon - dlon, rec.lon + dlon)).fetchall()

        new_phones = set(rec.phones or [])
        for row in rows:
            qx, qy = estimated_position(row["lon"], row["lat"], row["bearing_deg"],
                                        self.dedup_assumed_distance_m)
            d = _haversine_m(px, py, qx, qy)
            old_phones = {p.strip() for p in (row["phones"] or "").split(",") if p.strip()}
            if new_phones and old_phones and (new_phones & old_phones):
                if d <= self.dedup_phone_radius_m:
                    return row
                continue          # тот же телефон, но далеко — другой щит
            if d <= self.dedup_radius_m:
                return row
        return None

    def save(self, rec: BannerRecord) -> bool:
        """Вставляет запись. Возвращает False, если дубликат уже есть."""
        key = dedup_key(rec)
        if self.find_duplicate(rec) is not None:
            return False
        row = rec.to_row()
        self.conn.execute(
            """INSERT OR REPLACE INTO banners
               (banner_id, dedup_key, panoid, lon, lat, timestamp, bearing_deg,
                category, phones, text, address, score, full_image_path,
                crop_image_path, source_url)
               VALUES (:banner_id, :dedup_key, :panoid, :lon, :lat, :timestamp,
                :bearing_deg, :category, :phones, :text, :address, :score,
                :full_image_path, :crop_image_path, :source_url)""",
            {**row, "dedup_key": key},
        )
        self.conn.commit()
        return True

    def all(self, category: str | None = None) -> Iterator[sqlite3.Row]:
        if category:
            yield from self.conn.execute(
                "SELECT * FROM banners WHERE category = ? ORDER BY category", (category,))
        else:
            yield from self.conn.execute("SELECT * FROM banners ORDER BY category")

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM banners").fetchone()[0]

    # ---- состояние обхода (резюмируемость) -------------------------------
    def enqueue(self, oids) -> None:
        self.conn.executemany(
            "INSERT OR IGNORE INTO crawl_frontier(oid) VALUES (?)",
            [(o,) for o in oids])
        self.conn.commit()

    def next_oid(self) -> str | None:
        row = self.conn.execute(
            "SELECT oid FROM crawl_frontier WHERE done = 0 LIMIT 1").fetchone()
        return row[0] if row else None

    def mark_oid_done(self, oid: str) -> None:
        self.conn.execute("UPDATE crawl_frontier SET done = 1 WHERE oid = ?", (oid,))

    def is_visited(self, panoid: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM crawl_visited WHERE panoid = ? LIMIT 1", (panoid,)).fetchone() is not None

    def mark_visited(self, panoid: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO crawl_visited(panoid) VALUES (?)", (panoid,))

    def points_in_cells(self, cells: list[str]) -> list[tuple[float, float]]:
        if not cells:
            return []
        q = ",".join("?" * len(cells))
        rows = self.conn.execute(
            f"SELECT lon, lat FROM crawl_cells WHERE cell IN ({q})", cells).fetchall()
        return [(r[0], r[1]) for r in rows]

    def mark_cell(self, cell: str, lon: float, lat: float) -> None:
        self.conn.execute("INSERT OR IGNORE INTO crawl_cells(cell, lon, lat) VALUES (?,?,?)",
                          (cell, lon, lat))

    def frontier_pending(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM crawl_frontier WHERE done = 0").fetchone()[0]

    def visited_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM crawl_visited").fetchone()[0]

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
