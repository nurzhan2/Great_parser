from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HEADERS = [
    "ID",
    "Дата съёмки",
    "Адрес",
    "Категория",
    "Частное объявление",
    "Тип рекламодателя",
    "Тип предложения",
    "Застройщик / рекламодатель",
    "ЖК / объект",
    "Телефон",
    "Ненадёжный телефон",
    "Сайт",
    "Telegram",
    "Текст объявления",
    "Тип конструкции",
    "Недвижимость",
    "OCR",
    "Score",
    "Широта",
    "Долгота",
    "Панорама Яндекс",
    "Кроп на сервере",
]


def _get(obj, key, default=""):
    try:
        if hasattr(obj, key):
            value = getattr(obj, key)
        else:
            value = obj[key]
    except (KeyError, TypeError):
        value = default

    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(x) for x in value)

    if isinstance(value, bool):
        return "да" if value else "нет"

    return value


def _row(rec):
    timestamp = _get(rec, "timestamp")

    if timestamp:
        try:
            shot_date = datetime.fromtimestamp(
                int(timestamp),
                tz=timezone.utc,
            ).strftime("%Y-%m-%d")
        except (ValueError, TypeError, OSError):
            shot_date = ""
    else:
        shot_date = ""

    developer = (
        _get(rec, "developer")
        or _get(rec, "advertiser")
        or _get(rec, "brand")
    )

    return [
        _get(rec, "banner_id"),
        shot_date,
        _get(rec, "address"),
        _get(rec, "category"),
        _get(rec, "personal_ad"),
        _get(rec, "advertiser_type"),
        _get(rec, "offer_type"),
        developer,
        _get(rec, "complex_name"),
        _get(rec, "phones"),
        _get(rec, "phones_unreliable"),
        _get(rec, "sites"),
        _get(rec, "telegram"),
        _get(rec, "text"),
        _get(rec, "construction"),
        _get(rec, "is_realty"),
        _get(rec, "ocr_engine"),
        _get(rec, "score"),
        _get(rec, "lat"),
        _get(rec, "lon"),
        _get(rec, "source_url"),
        _get(rec, "crop_image_path"),
    ]


class GoogleSheetsSink:
    def __init__(
        self,
        spreadsheet_id: str,
        credentials_path: str,
        sheet_name: str = "Лиды",
    ):
        self.spreadsheet_id = spreadsheet_id
        self.sheet_name = sheet_name

        path = Path(credentials_path).expanduser()

        credentials = Credentials.from_service_account_file(
            str(path),
            scopes=SCOPES,
        )

        self.service = build(
            "sheets",
            "v4",
            credentials=credentials,
            cache_discovery=False,
        )

        self._ensure_sheet()
        self._ensure_header()

    @property
    def _quoted_sheet(self):
        return "'" + self.sheet_name.replace("'", "''") + "'"

    def _ensure_sheet(self):
        meta = self.service.spreadsheets().get(
            spreadsheetId=self.spreadsheet_id,
            fields="sheets.properties.title",
        ).execute()

        titles = {
            item["properties"]["title"]
            for item in meta.get("sheets", [])
        }

        if self.sheet_name not in titles:
            self.service.spreadsheets().batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={
                    "requests": [
                        {
                            "addSheet": {
                                "properties": {
                                    "title": self.sheet_name,
                                    "gridProperties": {
                                        "frozenRowCount": 1,
                                    },
                                }
                            }
                        }
                    ]
                },
            ).execute()

    def _ensure_header(self):
        result = self.service.spreadsheets().values().get(
            spreadsheetId=self.spreadsheet_id,
            range=f"{self._quoted_sheet}!A1:V1",
        ).execute()

        if not result.get("values"):
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self._quoted_sheet}!A1",
                valueInputOption="RAW",
                body={"values": [HEADERS]},
            ).execute()

    def append_many(self, records):
        records = list(records)

        if not records:
            return

        values = [_row(rec) for rec in records]

        try:
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self._quoted_sheet}!A:V",
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": values},
            ).execute()

            log.info(
                "Google Sheets: добавлено %d строк",
                len(records),
            )

        except Exception:
            # Google Sheets никогда не должен останавливать основной парсер.
            log.exception(
                "Google Sheets: ошибка синхронизации; "
                "записи сохранены в SQLite"
            )


def build_google_sheets(cfg):
    if not cfg.get("google_sheets.enabled", False):
        return None

    spreadsheet_id = (
        os.getenv("GOOGLE_SHEET_ID")
        or cfg.get("google_sheets.spreadsheet_id", "")
    )

    credentials_path = (
        os.getenv("GOOGLE_SHEETS_CREDENTIALS")
        or cfg.get("google_sheets.credentials_path", "")
    )

    sheet_name = cfg.get(
        "google_sheets.sheet_name",
        "Лиды",
    )

    if not spreadsheet_id:
        log.warning(
            "Google Sheets включён, но GOOGLE_SHEET_ID не задан"
        )
        return None

    if not credentials_path:
        log.warning(
            "Google Sheets включён, но credentials не заданы"
        )
        return None

    try:
        sink = GoogleSheetsSink(
            spreadsheet_id,
            credentials_path,
            sheet_name,
        )

        log.info(
            "Google Sheets подключён: лист «%s»",
            sheet_name,
        )

        return sink

    except Exception:
        log.exception(
            "Google Sheets не удалось подключить; "
            "парсер продолжит работу только с SQLite"
        )
        return None

