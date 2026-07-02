"""SQLite-хранилище баннеров с дедупликацией одного щита из разных панорам."""
from __future__ import annotations

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
    """Ключ дедупа: по телефонам (надёжнее всего), иначе по гео+азимуту."""
    if rec.phones:
        return "ph:" + "|".join(sorted(rec.phones))
    b = round((rec.bearing_deg or 0) / 10) * 10
    return f"geo:{round(rec.lon, 4)}:{round(rec.lat, 4)}:{b}:{rec.category}"


class Storage:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def save(self, rec: BannerRecord) -> bool:
        """Вставляет запись. Возвращает False, если дубликат уже есть."""
        key = dedup_key(rec)
        cur = self.conn.execute("SELECT 1 FROM banners WHERE dedup_key = ? LIMIT 1", (key,))
        if cur.fetchone():
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
