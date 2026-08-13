"""CLI: demo | crawl | grid | export | stats."""
from __future__ import annotations

import argparse
import logging
import sys

from . import runlog
from .config import Config
from .detect import RegionDetector
from .export import export_xlsx
from .geo import grid_seeds, road_seeds
from .pipeline import Pipeline

log = logging.getLogger(__name__)


def _pipeline(args) -> Pipeline:
    cfg = Config.load(args.config)
    p = Pipeline(cfg)
    if getattr(args, "region", None):
        p.detector = RegionDetector(args.region)   # ручные регионы поверх конфига
    return p


def cmd_demo(args) -> None:
    p = _pipeline(args)
    recs = p.process_point(args.lon, args.lat)
    for r in recs:
        print(f"  [{r.category}] {r.banner_id}  тел: {r.phones or '—'}  азимут {r.bearing_deg:.0f}°")
    print(f"Готово: {len(recs)} баннеров. Всего в БД: {p.storage.count()}")
    p.close()


def cmd_crawl(args) -> None:
    # Обход идёт часами — heartbeat нужен, чтобы после внезапной смерти
    # процесса было видно, на каком этапе и с какой памятью он был.
    runlog.start_heartbeat(args.heartbeat)
    p = _pipeline(args)
    total = 0
    try:
        for r in p.crawl(args.lon, args.lat, max_panoramas=args.limit):
            total += 1
            print(f"  [{r.category}] {r.panoid[:16]}… тел: {r.phones or '—'}  "
                  f"{r.address or ''}", flush=True)
    finally:
        # Итог печатаем даже при обрыве — иначе непонятно, сколько успели снять.
        log.info("итог обхода: +%d баннеров за запуск, всего в БД %d, "
                 "посещено панорам %d, в очереди %d",
                 total, p.storage.count(), p.storage.visited_count(),
                 p.storage.frontier_pending())
        print(f"Обход остановлен: +{total} баннеров. Всего в БД: {p.storage.count()}, "
              f"посещено панорам: {p.storage.visited_count()}, "
              f"в очереди: {p.storage.frontier_pending()}", flush=True)
        p.close()


def cmd_grid(args) -> None:
    p = _pipeline(args)
    bbox = tuple(args.bbox)
    seeds = road_seeds(bbox, args.step) if args.road else grid_seeds(bbox, args.step)
    total, pts = 0, 0
    for lon, lat in seeds:
        pts += 1
        total += len(p.process_point(lon, lat))
    print(f"Обработано точек: {pts}, новых баннеров: {total}. Всего в БД: {p.storage.count()}")
    p.close()


def cmd_export(args) -> None:
    cfg = Config.load(args.config)
    from .storage import Storage
    st = Storage(cfg.get("storage.db_path", "data/banners.sqlite"))
    out = args.out or cfg.get("export.xlsx_path", "data/banners.xlsx")
    # Фильтр темы применяется ЗДЕСЬ, а не при сборе: собранное можно
    # отфильтровать, несобранное — уже нет. Приоритет: --all > --category > конфиг.
    category = None if args.all else (args.category
                                      or cfg.get("filter.only_category", None))
    total = st.count()
    n = export_xlsx(st, out, category=category)
    log.info("выгрузка: %d строк из %d в БД (фильтр темы: %s)",
             n, total, category or "нет")
    print(f"Выгружено строк: {n} из {total} в БД → {out}")
    if category and n < total:
        print(f"  (отфильтровано по теме '{category}'; --all выгрузит всё)")
    st.close()


def cmd_stats(args) -> None:
    cfg = Config.load(args.config)
    from .storage import Storage
    st = Storage(cfg.get("storage.db_path", "data/banners.sqlite"))
    print(f"Всего баннеров: {st.count()}")
    rows = st.conn.execute(
        "SELECT category, COUNT(*) c FROM banners GROUP BY category ORDER BY c DESC")
    for cat, c in rows:
        print(f"  {cat}: {c}")
    st.close()


def _common_options() -> argparse.ArgumentParser:
    """Опции, работающие и до, и после подкоманды.

    default=SUPPRESS обязателен: подкоманда парсит аргументы в отдельный
    namespace и копирует его поверх основного, поэтому обычный default затёр бы
    значение, заданное перед подкомандой. SUPPRESS просто не кладёт ключ.

    По той же причине здесь нельзя вызывать set_defaults(): parents=[...]
    переиспользует те же самые объекты actions, и set_defaults на главном
    парсере подменил бы SUPPRESS у подкоманд. Значения по умолчанию
    проставляет _apply_defaults() уже после разбора.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", default=argparse.SUPPRESS, help="путь к config.yaml")
    p.add_argument("--log", default=argparse.SUPPRESS, metavar="FILE",
                   help="дублировать лог в файл (пишется с flush на каждой строке)")
    p.add_argument("--log-level", default=argparse.SUPPRESS,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="уровень лога")
    p.add_argument("--heartbeat", type=float, default=argparse.SUPPRESS, metavar="SEC",
                   help="период строки «жив: этап…» при обходе, 0 — выключить")
    return p


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    ap = argparse.ArgumentParser("banner_parser", parents=[common],
                                 description="Парсер наружной рекламы из Яндекс.Панорам")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", parents=[common], help="обработать одну точку")
    d.add_argument("--lon", type=float, required=True)
    d.add_argument("--lat", type=float, required=True)
    d.add_argument("--region", type=float, nargs=4, action="append",
                   metavar=("FX0", "FY0", "FX1", "FY1"),
                   help="ручной регион баннера (можно несколько раз)")
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("crawl", parents=[common], help="резюмируемый обход графа от точки (в пределах bbox)")
    c.add_argument("--lon", type=float, required=True)
    c.add_argument("--lat", type=float, required=True)
    c.add_argument("--limit", type=int, default=None,
                   help="макс. панорам за запуск (по умолчанию — без лимита)")
    c.set_defaults(func=cmd_crawl)

    g = sub.add_parser("grid", parents=[common], help="обход bbox по сетке/дорогам")
    g.add_argument("--bbox", type=float, nargs=4, required=True,
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    g.add_argument("--step", type=float, default=150.0, help="шаг seed-точек, м")
    g.add_argument("--road", action="store_true", help="сажать точки на дороги (osmnx)")
    g.set_defaults(func=cmd_grid)

    e = sub.add_parser("export", parents=[common], help="выгрузка БД в Excel")
    e.add_argument("--out", default=None)
    e.add_argument("--category", default=None,
                   help="фильтр по теме (по умолчанию — filter.only_category из конфига)")
    e.add_argument("--all", action="store_true",
                   help="выгрузить всё, игнорируя filter.only_category")
    e.set_defaults(func=cmd_export)

    s = sub.add_parser("stats", parents=[common], help="статистика по БД")
    s.set_defaults(func=cmd_stats)
    return ap


_DEFAULTS = {"config": None, "log": None, "log_level": "INFO", "heartbeat": 60.0}


def _apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Проставить общие опции, которых нет в namespace (см. SUPPRESS выше)."""
    for key, value in _DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def main() -> None:
    args = _apply_defaults(build_parser().parse_args())

    # Порядок важен: сначала лог и перехватчики, потом шапка, и только потом
    # тяжёлые импорты внутри команд. Иначе смерть на загрузке модели
    # не оставит в логе вообще ничего — ровно так и было раньше.
    log_path = runlog.setup_logging(args.log, args.log_level)
    runlog.install_crash_handlers()
    runlog.log_startup(cfg_path=args.config, log_path=log_path)

    try:
        args.func(args)
    except Exception:                       # noqa: BLE001 — нужен диагноз, не стектрейс в никуда
        runlog.set_stage("аварийное завершение")
        log.critical("=== ИСКЛЮЧЕНИЕ в команде «%s» ===", args.cmd, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
