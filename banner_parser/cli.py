"""CLI: demo | crawl | grid | export | stats."""
from __future__ import annotations

import argparse
import logging

from .config import Config
from .detect import RegionDetector
from .export import export_xlsx
from .geo import grid_seeds, road_seeds
from .pipeline import Pipeline


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
    p = _pipeline(args)
    total = 0
    for r in p.crawl(args.lon, args.lat, max_panoramas=args.limit):
        total += 1
        print(f"  [{r.category}] {r.panoid[:16]}… тел: {r.phones or '—'}  {r.address or ''}")
    print(f"Обход остановлен: +{total} баннеров. Всего в БД: {p.storage.count()}, "
          f"посещено панорам: {p.storage.visited_count()}, в очереди: {p.storage.frontier_pending()}")
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
    n = export_xlsx(st, out, category=args.category)
    print(f"Выгружено строк: {n} → {out}")
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("banner_parser", description="Парсер наружной рекламы из Яндекс.Панорам")
    ap.add_argument("--config", default=None, help="путь к config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="обработать одну точку")
    d.add_argument("--lon", type=float, required=True)
    d.add_argument("--lat", type=float, required=True)
    d.add_argument("--region", type=float, nargs=4, action="append",
                   metavar=("FX0", "FY0", "FX1", "FY1"),
                   help="ручной регион баннера (можно несколько раз)")
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("crawl", help="резюмируемый обход графа от точки (в пределах bbox)")
    c.add_argument("--lon", type=float, required=True)
    c.add_argument("--lat", type=float, required=True)
    c.add_argument("--limit", type=int, default=None,
                   help="макс. панорам за запуск (по умолчанию — без лимита)")
    c.set_defaults(func=cmd_crawl)

    g = sub.add_parser("grid", help="обход bbox по сетке/дорогам")
    g.add_argument("--bbox", type=float, nargs=4, required=True,
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    g.add_argument("--step", type=float, default=150.0, help="шаг seed-точек, м")
    g.add_argument("--road", action="store_true", help="сажать точки на дороги (osmnx)")
    g.set_defaults(func=cmd_grid)

    e = sub.add_parser("export", help="выгрузка БД в Excel")
    e.add_argument("--out", default=None)
    e.add_argument("--category", default=None, help="фильтр по теме")
    e.set_defaults(func=cmd_export)

    s = sub.add_parser("stats", help="статистика по БД")
    s.set_defaults(func=cmd_stats)
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "transformers", "PIL", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
