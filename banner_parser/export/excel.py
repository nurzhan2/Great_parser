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
# Колонки сегмента «наружная реклама недвижимости»: где стоит щит ->
# что за конструкция -> чей он -> какое предложение -> как связаться.
# Контакты РАЗВЕДЕНЫ по источнику: прочитанное со щита и взятое из
# справочника — данные разной надёжности.
COLUMNS = [
    ("crop", "Фото щита", 58),
    ("address", "Адрес щита", 30),
    ("construction", "Тип конструкции", 18),
    ("developer", "Застройщик", 20),
    ("complex_name", "ЖК", 22),
    ("offer_type", "Тип предложения", 18),
    ("advertiser_type", "Тип рекламодателя", 18),
    ("phones", "Телефон СО ЩИТА", 18),
    ("dir_phone", "Телефон из справочника", 20),
    ("sites", "Сайт СО ЩИТА", 18),
    ("dir_site", "Сайт из справочника", 20),
    ("phones_unreliable", "Ненадёжные контакты", 18),
    ("shot_date", "Дата съёмки", 13),
    ("lat", "Широта", 11),
    ("lon", "Долгота", 11),
    ("source_url", "Панорама", 12),
    ("text", "Текст со щита", 30),
    ("banner_id", "ID", 14),
]

_ROW_H = 210            # высота строки под картинку, px
_IMG_W, _IMG_H = 440, 200


def _fit(path: str, box_w: int, box_h: int) -> tuple[float, float]:
    """Масштаб картинки под ячейку, сохраняя пропорции."""
    with Image.open(path) as im:
        w, h = im.size
    return min(box_w / w, box_h / h, 1.0), (w, h)


def export_xlsx(storage: Storage, out_path: str, category=None,
                realty_only: bool = False, include_unsure: bool = True) -> int:
    """realty_only=True выгружает только сегмент недвижимости; частные
    объявления при этом уходят на ОТДЕЛЬНЫЙ лист и в общую базу контактов
    не попадают — это физлица, а не рекламодатели рынка."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(out_path, {"constant_memory": False})
    header = wb.add_format({"bold": True, "bg_color": "#1F3864", "font_color": "white",
                            "align": "center", "valign": "vcenter", "border": 1})
    cell = wb.add_format({"valign": "vcenter", "text_wrap": True, "border": 1})
    warn = wb.add_format({"valign": "vcenter", "text_wrap": True, "border": 1,
                          "font_color": "#9C0006", "bg_color": "#FFC7CE"})
    dirf = wb.add_format({"valign": "vcenter", "text_wrap": True, "border": 1,
                          "italic": True, "font_color": "#3F5F8F"})

    if realty_only:
        sheets = [("недвижимость", list(storage.realty(include_unsure, personal=False))),
                  ("частные объявления", list(storage.realty(include_unsure, personal=True)))]
    else:
        sheets = [("banners", list(storage.all(category)))]

    total = 0
    for title, rows in sheets:
        ws = wb.add_worksheet(title)
        for c, (_, name, width) in enumerate(COLUMNS):
            ws.set_column(c, c, width)
            ws.write(0, c, name, header)
        ws.set_row(0, 24)
        ws.freeze_panes(1, 0)
        col = {k: i for i, (k, _, _) in enumerate(COLUMNS)}

        for r, row in enumerate(rows, start=1):
            ws.set_row(r, _ROW_H)
            keys = row.keys()

            def v(k):
                return (row[k] if k in keys else None) or ""

            dev = v("developer") or v("advertiser")
            matched = (row["brand_matched"] if "brand_matched" in keys else 0)
            if dev and not matched:
                dev = str(dev) + " (не опознан)"

            ws.write(r, col["address"], v("address"), cell)
            ws.write(r, col["construction"], v("construction"), cell)
            ws.write(r, col["developer"], dev, cell)
            ws.write(r, col["complex_name"], v("complex_name"), cell)
            ws.write(r, col["offer_type"], v("offer_type"), cell)
            ws.write(r, col["advertiser_type"], v("advertiser_type"), cell)
            ws.write(r, col["phones"], v("phones"), cell)
            ws.write(r, col["dir_phone"], v("dir_phone"), dirf)
            ws.write(r, col["sites"], v("sites"), cell)
            ws.write(r, col["dir_site"], v("dir_site"), dirf)
            ws.write(r, col["phones_unreliable"], v("phones_unreliable"), warn)
            ts = row["timestamp"] if "timestamp" in keys else None
            ws.write(r, col["shot_date"],
                     datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "", cell)
            ws.write(r, col["lat"], row["lat"], cell)
            ws.write(r, col["lon"], row["lon"], cell)
            ws.write(r, col["text"], v("text"), cell)
            ws.write(r, col["banner_id"], v("banner_id"), cell)
            if row["source_url"]:
                ws.write_url(r, col["source_url"], row["source_url"], cell, "смотреть")
            _embed(ws, r, col["crop"], row["crop_image_path"], cell)

        ws.autofilter(0, 1, max(1, len(rows)), len(COLUMNS) - 1)
        log.info("лист «%s»: %d строк", title, len(rows))
        if title != "частные объявления":
            total += len(rows)

    wb.close()
    log.info("export: %d строк -> %s", total, out_path)
    return total


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
