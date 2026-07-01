"""Выгрузка баннеров в Excel со встроенными фото (панорама + кроп)."""
from __future__ import annotations

import logging
from pathlib import Path

import xlsxwriter
from PIL import Image

from ..storage import Storage

log = logging.getLogger(__name__)

COLUMNS = [
    ("crop", "Баннер", 62),
    ("category", "Тема", 14),
    ("phones", "Телефоны", 20),
    ("address", "Адрес", 28),
    ("text", "Распознанный текст", 40),
    ("bearing_deg", "Азимут°", 9),
    ("lon", "Долгота", 12),
    ("lat", "Широта", 12),
    ("source_url", "Ссылка", 26),
    ("banner_id", "ID", 16),
]

_ROW_H = 210            # высота строки под картинку, px
_IMG_W, _IMG_H = 440, 200


def _fit(path: str, box_w: int, box_h: int) -> tuple[float, float]:
    """Масштаб картинки под ячейку, сохраняя пропорции."""
    with Image.open(path) as im:
        w, h = im.size
    return min(box_w / w, box_h / h, 1.0), (w, h)


def export_xlsx(storage: Storage, out_path: str, category: str | None = None) -> int:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(out_path, {"constant_memory": False})
    ws = wb.add_worksheet("banners")
    header = wb.add_format({"bold": True, "bg_color": "#1F3864", "font_color": "white",
                            "align": "center", "valign": "vcenter", "border": 1})
    cell = wb.add_format({"valign": "vcenter", "text_wrap": True, "border": 1})

    for c, (_, title, width) in enumerate(COLUMNS):
        ws.set_column(c, c, width)
        ws.write(0, c, title, header)
    ws.set_row(0, 24)
    ws.freeze_panes(1, 0)

    col_idx = {key: i for i, (key, _, _) in enumerate(COLUMNS)}
    n = 0
    for r, row in enumerate(storage.all(category), start=1):
        ws.set_row(r, _ROW_H)
        ws.write(r, col_idx["banner_id"], row["banner_id"], cell)
        ws.write(r, col_idx["category"], row["category"], cell)
        ws.write(r, col_idx["phones"], row["phones"] or "", cell)
        ws.write(r, col_idx["address"], row["address"] or "", cell)
        ws.write(r, col_idx["bearing_deg"],
                 round(row["bearing_deg"], 1) if row["bearing_deg"] is not None else "", cell)
        ws.write(r, col_idx["lon"], row["lon"], cell)
        ws.write(r, col_idx["lat"], row["lat"], cell)
        ws.write(r, col_idx["text"], row["text"] or "", cell)
        if row["source_url"]:
            ws.write_url(r, col_idx["source_url"], row["source_url"], cell, "открыть")
        else:
            ws.write(r, col_idx["source_url"], "", cell)
        _embed(ws, r, col_idx["crop"], row["crop_image_path"], cell)
        n += 1

    wb.close()
    log.info("export: %d rows -> %s", n, out_path)
    return n


def _embed(ws, r: int, c: int, path: str | None, cell) -> None:
    if not path or not Path(path).exists():
        ws.write(r, c, "", cell)
        return
    scale, _ = _fit(path, _IMG_W, _IMG_H)
    ws.write(r, c, "", cell)
    ws.insert_image(r, c, path, {
        "x_scale": scale, "y_scale": scale,
        "x_offset": 4, "y_offset": 4, "object_position": 1,
    })
