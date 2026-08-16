"""Выгрузка баннеров в Excel со встроенными фото (панорама + кроп)."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import xlsxwriter
from PIL import Image

from ..storage import Storage

log = logging.getLogger(__name__)

# Порядок под задачу продавца рекламных мест: где стоит -> что за
# конструкция -> кто висит -> как связаться. Контакты РАЗВЕДЕНЫ по источнику:
# прочитанное со щита и взятое из справочника — данные разной надёжности,
# и смешивать их в одной колонке нельзя.
COLUMNS = [
    ("crop", "Фото щита", 58),
    ("address", "Адрес", 30),
    ("construction", "Тип конструкции", 18),
    ("brand", "Рекламодатель", 22),
    ("category", "Категория", 14),
    ("phones", "Телефон СО ЩИТА", 18),
    ("sites", "Сайт СО ЩИТА", 20),
    ("dir_phone", "Телефон из справочника", 20),
    ("dir_site", "Сайт из справочника", 20),
    ("phones_unreliable", "Ненадёжные контакты", 20),
    ("shot_date", "Дата съёмки", 13),
    ("lat", "Широта", 11),
    ("lon", "Долгота", 11),
    ("source_url", "Панорама", 13),
    ("text", "Текст со щита", 34),
    ("banner_id", "ID", 14),
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
    warn = wb.add_format({"valign": "vcenter", "text_wrap": True, "border": 1,
                          "font_color": "#9C0006", "bg_color": "#FFC7CE"})
    dir_fmt = wb.add_format({"valign": "vcenter", "text_wrap": True, "border": 1,
                             "italic": True, "font_color": "#3F5F8F"})
    n = 0
    for r, row in enumerate(storage.all(category), start=1):
        ws.set_row(r, _ROW_H)
        keys = row.keys()

        def val(k):
            return (row[k] if k in keys else None) or ""

        # Бренд: каноничный из справочника, иначе — как прочитано со щита,
        # с пометкой, что он не опознан.
        brand = val("brand") or val("advertiser")
        matched = bool(row["brand_matched"]) if "brand_matched" in keys else False
        if brand and not matched:
            brand = f"{brand} (не опознан)"

        ws.write(r, col_idx["address"], val("address"), cell)
        ws.write(r, col_idx["construction"], val("construction"), cell)
        ws.write(r, col_idx["brand"], brand, cell)
        ws.write(r, col_idx["category"], val("category"), cell)
        ws.write(r, col_idx["phones"], val("phones"), cell)
        ws.write(r, col_idx["sites"], val("sites"), cell)
        ws.write(r, col_idx["dir_phone"], val("dir_phone"), dir_fmt)
        ws.write(r, col_idx["dir_site"], val("dir_site"), dir_fmt)
        ws.write(r, col_idx["phones_unreliable"], val("phones_unreliable"), warn)
        ts = row["timestamp"] if "timestamp" in keys else None
        ws.write(r, col_idx["shot_date"],
                 datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "", cell)
        ws.write(r, col_idx["lat"], row["lat"], cell)
        ws.write(r, col_idx["lon"], row["lon"], cell)
        ws.write(r, col_idx["text"], val("text"), cell)
        ws.write(r, col_idx["banner_id"], val("banner_id"), cell)
        if row["source_url"]:
            ws.write_url(r, col_idx["source_url"], row["source_url"], cell, "смотреть")
        else:
            ws.write(r, col_idx["source_url"], "", cell)
        _embed(ws, r, col_idx["crop"], row["crop_image_path"], cell)
        n += 1

    ws.autofilter(0, 1, max(1, n), len(COLUMNS) - 1)
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
