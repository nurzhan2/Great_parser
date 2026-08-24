#!/usr/bin/env python3
"""Синхронизация записей из SQLite в Google Sheets.

Заливает те записи, которых в таблице ещё нет (сверка по banner_id).
Годится и для первичной заливки, и для догона пропущенного,
если Google был недоступен во время обхода.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banner_parser.config import Config
from banner_parser.export.google_sheets import build_google_sheets, _row

log = logging.getLogger("sync_sheets")

BATCH = 200          # строк за один запрос к API
PAUSE = 1.5          # пауза между батчами: квота 60 запросов/мин


def existing_ids(sink) -> set[str]:
    """ID, уже лежащие в таблице — первая колонка листа."""
    res = sink.service.spreadsheets().values().get(
        spreadsheetId=sink.spreadsheet_id,
        range=f"{sink._quoted_sheet}!A2:A",
    ).execute()
    return {r[0] for r in res.get("values", []) if r}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--db", default=None, help="путь к SQLite (по умолчанию из конфига)")
    ap.add_argument("--dry-run", action="store_true", help="показать, но не писать")
    ap.add_argument("--limit", type=int, default=0, help="залить не больше N записей")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = Config.load(args.config)
    db_path = args.db or cfg.get("storage.db_path", "data/banners.sqlite")

    sink = build_google_sheets(cfg)
    if sink is None:
        log.error("Google Sheets не подключён — проверь переменные окружения")
        return 2

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM banners ORDER BY rowid")]
    conn.close()
    log.info("в базе %s: %d записей", db_path, len(rows))

    have = existing_ids(sink)
    log.info("в таблице уже: %d записей", len(have))

    todo = [r for r in rows if str(r.get("banner_id", "")) not in have]
    if args.limit:
        todo = todo[: args.limit]

    log.info("к заливке: %d записей", len(todo))
    if not todo:
        log.info("нечего заливать — таблица актуальна")
        return 0

    if args.dry_run:
        log.info("--dry-run: ничего не записано")
        for r in todo[:5]:
            log.info("  пример: %s | %s | %s",
                     r.get("banner_id"), (r.get("address") or "")[:40], r.get("category"))
        if len(todo) > 5:
            log.info("  ... и ещё %d", len(todo) - 5)
        return 0

    sent = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        values = [_row(r) for r in chunk]
        for attempt in range(1, 4):
            try:
                sink.service.spreadsheets().values().append(
                    spreadsheetId=sink.spreadsheet_id,
                    range=f"{sink._quoted_sheet}!A:V",
                    valueInputOption="RAW",
                    insertDataOption="INSERT_ROWS",
                    body={"values": values},
                ).execute()
                sent += len(chunk)
                log.info("залито %d/%d", sent, len(todo))
                break
            except Exception as exc:
                if attempt == 3:
                    log.error("батч не залит после 3 попыток: %s", type(exc).__name__)
                    return 1
                wait = 2 ** attempt
                log.warning("ошибка (%s), повтор через %ds", type(exc).__name__, wait)
                time.sleep(wait)
        time.sleep(PAUSE)

    log.info("готово: залито %d записей", sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
