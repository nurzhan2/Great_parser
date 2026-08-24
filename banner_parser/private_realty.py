"""Режим private_realty_crawl: обход панорам по улицам малоэтажной и частной
застройки вместо магистралей.

Отдельный модуль — намеренно. Ни один существующий файл проекта не изменяется:
детекция, OWLv2, промпты, margin-фильтр, verify, OCR (включая лимиты облачного),
дедуп, SQLite, Google Sheets и export работают ровно как раньше. Всё нужное
берётся импортом:

  * дорожный граф   — geo.seed._graph_from_bbox / _geo_len_m
  * рамка и сетка   — pipeline._in_bbox / _cell_coords / _neighbor_keys / _too_close
  * сам конвейер    — Pipeline.process_panorama (он же пишет в Sheets)

Отличие от crawl() ровно одно: источник точек. Тот идёт по графу соседних
панорам и утекает на ближайшую магистраль; этот сажает точки на выбранные
типы дорог OSM.

Запуск:
    python3 -u -m banner_parser.private_realty --bbox 37.69 55.64 37.71 55.65 --dry-run
    python3 -u -m banner_parser.private_realty --bbox 37.69 55.64 37.71 55.65 --limit 5
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter

from . import runlog
from .config import Config

log = logging.getLogger(__name__)

# Порядок = порядок приоритета. Магистрали (motorway/trunk/primary/secondary/
# tertiary) сюда намеренно не входят: режим делается ради объявлений «продам
# дом» на заборах, а обход по шоссе уводит от них.
# service последним: это и внутриквартальные проезды (нужны), и парковочные
# аллеи (панорам там обычно нет).
PRIVATE_ROAD_CLASSES = ("living_street", "residential", "unclassified", "service")


# ---- приоритет типов дорог ----------------------------------------------
def _highway_name(v) -> str:
    """Значение тега highway у ребра. У одного OSM-way значений может быть
    несколько — osmnx кладёт их списком; берём первое строковое."""
    if isinstance(v, (list, tuple, set)):
        for item in v:
            if isinstance(item, str):
                return item
        return "?"
    return v if isinstance(v, str) else "?"


def _fmt(c: Counter) -> str:
    return ", ".join(f"{k}={v}" for k, v in c.most_common()) or "пусто"


def _ordered_geoms(edges_ll, priority: tuple) -> list:
    """Геометрии рёбер: отфильтрованы по типам и упорядочены по приоритету.

    Фильтр Overpass в _graph_from_bbox задан регуляркой без якорей и иногда
    протаскивает лишнее, а порядок обхода он не задаёт вовсе — досортировываем
    и дочищаем локально.
    """
    geoms = list(edges_ll.geometry)
    if "highway" not in edges_ll.columns:
        log.warning("в дорожном графе нет колонки highway — приоритет типов "
                    "не применён, идём по всем %d рёбрам", len(geoms))
        return geoms

    kinds = [_highway_name(v) for v in edges_ll["highway"]]
    ranked = [(priority.index(k), i) for i, k in enumerate(kinds) if k in priority]
    ranked.sort()                       # стабильно: (приоритет, исходный порядок)

    log.info("дороги: найдено рёбер %d (%s)", len(geoms), _fmt(Counter(kinds)))
    log.info("дороги: после фильтра типов осталось %d (%s)", len(ranked),
             _fmt(Counter(kinds[i] for _, i in ranked)))
    log.info("дороги: приоритет типов %s", " > ".join(priority))
    dropped = Counter(k for k in kinds if k not in priority)
    if dropped:
        log.info("дороги: отброшено по типу %d (%s)",
                 sum(dropped.values()), _fmt(dropped))
    if not ranked:
        log.warning("после фильтра не осталось ни одного ребра — проверьте "
                    "--road-class и рамку --bbox")
    return [geoms[i] for _, i in ranked]


def _graph(ox, bbox: tuple, road_classes: tuple):
    """Дорожный граф по типам highway.

    Своя обёртка вместо geo.seed._graph_from_bbox нужна ровно из-за двух
    параметров, которых там нет, — и оба здесь принципиальны:

      retain_all=True     osmnx по умолчанию оставляет только КРУПНЕЙШУЮ
                          связную компоненту. Сеть магистралей связна, и для
                          обычного road_seeds это незаметно. Сеть жилых улиц
                          без магистралей рассыпается на острова, и умолчание
                          выбрасывает почти всё: замер на Мещерском посёлке —
                          76 рёбер против 782, residential 6 против 115.
      truncate_by_edge    иначе улица, чуть выходящая за рамку, теряется
                          целиком вместе с попавшей в рамку частью.

    Регулярка фильтра построена как в geo.seed — без якорей, намеренно:
    так же ловятся производные вроде living_street.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    custom = '["highway"~"' + "|".join(road_classes) + '"]'
    try:                                # osmnx >= 2.0
        return ox.graph_from_bbox(bbox=(min_lon, min_lat, max_lon, max_lat),
                                  custom_filter=custom,
                                  retain_all=True, truncate_by_edge=True)
    except TypeError:                   # osmnx 1.x: четыре позиционных аргумента
        log.info("osmnx со старой сигнатурой graph_from_bbox — используем её")
        return ox.graph_from_bbox(max_lat, min_lat, max_lon, min_lon,
                                  custom_filter=custom,
                                  retain_all=True, truncate_by_edge=True)


# ---- seed-точки вдоль выбранных улиц ------------------------------------
def road_seeds_private(bbox, step_m=60.0, road_classes=PRIVATE_ROAD_CLASSES):
    """Точки вдоль улиц заданных типов, в порядке приоритета этих типов.

    Отката к grid_seeds здесь сознательно НЕТ. В road_seeds() он уместен:
    там любая точка на дороге годится. Здесь равномерная сетка попадёт в том
    числе на магистрали — ровно то, от чего режим и уходит. Лучше пустой
    результат с внятной причиной, чем тихий обход не тех улиц.
    """
    from .geo.seed import _geo_len_m

    bbox = tuple(bbox)
    road_classes = tuple(road_classes)
    try:
        import osmnx as ox
    except Exception as e:      # noqa: BLE001
        log.error("osmnx недоступен (%s). Режим private_realty_crawl без него "
                  "работать не может: точки сажаются на дорожную сеть OSM", e)
        return

    runlog.set_stage("запрос дорожной сети OSM")
    t0 = time.monotonic()
    try:
        G = _graph(ox, bbox, road_classes)
    except Exception as e:      # noqa: BLE001
        log.error("не удалось построить дорожный граф (%s: %s). Частая причина — "
                  "слишком большая рамка: жилых улиц и проездов на порядок больше, "
                  "чем магистралей, и Overpass отваливается по таймауту. "
                  "Сузьте --bbox и повторите", type(e).__name__, e)
        return
    log.info("дорожный граф построен за %.1f с, рёбер %d, запрошенные классы %s",
             time.monotonic() - t0, len(G.edges), ", ".join(road_classes))

    G = ox.project_graph(G)
    _, edges = ox.graph_to_gdfs(G)
    edges_ll = edges.to_crs(epsg=4326)
    geoms = _ordered_geoms(edges_ll, road_classes)
    if not geoms:
        return

    total_m = sum(_geo_len_m(g) for g in geoms)
    log.info("дороги: суммарная длина %.2f км, seed-точек ожидается ≈%d (шаг %s м)",
             total_m / 1000.0, int(total_m / step_m) + len(geoms), step_m)

    seen: set[tuple[float, float]] = set()
    for geom in geoms:
        if geom.length == 0:
            continue
        n = max(1, int(_geo_len_m(geom) / step_m))
        for k in range(n + 1):
            p = geom.interpolate(k / n, normalized=True)
            key = (round(p.x, 5), round(p.y, 5))
            if key not in seen:
                seen.add(key)
                yield p.x, p.y


# ---- обход ---------------------------------------------------------------
def crawl_roads(p, bbox, step_m, road_classes, max_points=None):
    """Обход по seed-точкам вдоль улиц. p — готовый Pipeline.

    Сетка ячеек min_distance_m, отметка посещённых панорам и весь конвейер —
    те же, что в Pipeline.crawl(); берутся импортом, а не копией.
    """
    from .pipeline import _cell_coords, _in_bbox, _neighbor_keys, _too_close
    from .yandex import Panorama

    min_dist = p.cfg.get("crawl.min_distance_m", 250)
    st = p.storage

    log.info("обход по улицам: bbox=%s, шаг seed-точек %s м, min_distance %s м, "
             "лимит за запуск %s", bbox, step_m, min_dist, max_points or "нет")
    log.info("типы дорог (в порядке приоритета): %s", " > ".join(road_classes))
    log.info("состояние до старта: посещено панорам %d, баннеров в БД %d",
             st.visited_count(), st.count())

    seeds = skip_bbox = skip_near = skip_nopano = processed = 0
    t_start = time.monotonic()
    try:
        for lon, lat in road_seeds_private(bbox, step_m, road_classes):
            seeds += 1
            if not _in_bbox(lon, lat, bbox):
                skip_bbox += 1
                continue
            # Та же сетка, что в crawl(): один щит не снимается повторно
            # с соседних точек одной улицы.
            clat, clon = _cell_coords(lon, lat, min_dist)
            if _too_close(lon, lat, st.points_in_cells(_neighbor_keys(clat, clon)),
                          min_dist):
                skip_near += 1
                continue
            runlog.set_stage(f"meta-запрос точки {lon:.4f},{lat:.4f}")
            raw = p.meta.by_coords(lon, lat)
            if raw is None:
                skip_nopano += 1
                continue
            ref = p.meta.parse(raw)
            if st.is_visited(ref.panoid):
                skip_nopano += 1
                st.commit()
                continue
            st.mark_visited(ref.panoid)
            if not _in_bbox(ref.lon, ref.lat, bbox):
                skip_bbox += 1
                st.commit()
                continue
            st.mark_cell(f"{clat}:{clon}", ref.lon, ref.lat)
            try:
                yield from p.process_panorama(Panorama(ref, p.http, p.workers))
            except Exception:   # noqa: BLE001 — обход не должен падать на одной точке
                log.exception("ошибка обработки панорамы %s (%.5f,%.5f) — пропускаем",
                              ref.panoid, ref.lon, ref.lat)
            st.commit()
            processed += 1
            if processed % 20 == 0:
                rate = processed / max(1e-9, (time.monotonic() - t_start) / 60)
                log.info("обход по улицам: seed-точек %d, снято %d (%.1f точек/мин), "
                         "пропущено близко %d / без панорамы %d / вне рамки %d, "
                         "баннеров %d, rss %s",
                         seeds, processed, rate, skip_near, skip_nopano,
                         skip_bbox, st.count(), runlog.rss_str())
            if max_points and processed >= max_points:
                log.info("достигнут лимит %d обработанных точек за запуск", max_points)
                break
    finally:
        # Итог печатаем и при обрыве: иначе непонятно, сколько успели снять.
        st.commit()
        log.info("итог обхода по улицам: seed-точек просмотрено %d, реально "
                 "обработано панорам %d; пропущено: ближе %s м — %d, без панорамы "
                 "или уже посещали — %d, вне рамки — %d",
                 seeds, processed, min_dist, skip_near, skip_nopano, skip_bbox)


# ---- CLI -----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python3 -m banner_parser.private_realty",
        description="Обход панорам по улицам малоэтажной и частной застройки")
    ap.add_argument("--config", default=None, help="путь к config.yaml")
    ap.add_argument("--log", default=None, metavar="FILE", help="дублировать лог в файл")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--heartbeat", type=float, default=60.0, metavar="SEC",
                    help="период строки «жив: этап…», 0 — выключить")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                    help="рамка обхода; по умолчанию crawl.bbox из конфига")
    ap.add_argument("--step", type=float, default=None, metavar="M",
                    help="шаг seed-точек вдоль улицы, м (по умолчанию 60)")
    ap.add_argument("--road-class", action="append", default=None, metavar="HIGHWAY",
                    help="тип дороги OSM; повторяемый, порядок = приоритет. "
                         "По умолчанию: " + ", ".join(PRIVATE_ROAD_CLASSES))
    ap.add_argument("--limit", type=int, default=None,
                    help="макс. панорам за запуск (по умолчанию — без лимита)")
    ap.add_argument("--dry-run", action="store_true",
                    help="только посчитать дороги и точки; модели не грузятся, "
                         "панорамы не запрашиваются, БД не трогается")
    return ap


def _run(args) -> None:
    cfg = Config.load(args.config)
    classes = tuple(args.road_class
                    or cfg.get("private_realty.road_classes",
                               list(PRIVATE_ROAD_CLASSES)))
    step = float(args.step if args.step is not None
                 else cfg.get("private_realty.step_m", 60.0))
    bbox = args.bbox or cfg.get("crawl.bbox", None)
    if not bbox:
        raise SystemExit("не задан bbox: укажите --bbox или crawl.bbox в config.yaml")
    bbox = tuple(bbox)

    if args.dry_run:
        # Считаем точки, не поднимая Pipeline: OWLv2 грузится десятки секунд
        # и занимает память, а для проверки выбора улиц он не нужен.
        n = sum(1 for _ in road_seeds_private(bbox, step, classes))
        print(f"Сухой прогон: seed-точек {n}. Панорамы не запрашивались, "
              f"БД не изменялась.", flush=True)
        return

    runlog.start_heartbeat(args.heartbeat)
    from .pipeline import Pipeline
    p = Pipeline(cfg)
    total = 0
    try:
        for r in crawl_roads(p, bbox, step, classes, args.limit):
            total += 1
            print(f"  [{r.category}] {r.panoid[:16]}… тел: {r.phones or '—'}  "
                  f"{r.address or ''}", flush=True)
    finally:
        log.info("итог: +%d баннеров за запуск, всего в БД %d, посещено панорам %d",
                 total, p.storage.count(), p.storage.visited_count())
        print(f"Обход по улицам остановлен: +{total} баннеров. "
              f"Всего в БД: {p.storage.count()}, "
              f"посещено панорам: {p.storage.visited_count()}", flush=True)
        p.close()


def main() -> None:
    args = build_parser().parse_args()
    # Порядок как в cli.py: сначала лог и перехватчики, потом шапка, и только
    # потом тяжёлые импорты — иначе смерть на загрузке модели не оставит следа.
    log_path = runlog.setup_logging(args.log, args.log_level)
    runlog.install_crash_handlers()
    runlog.log_startup(cfg_path=args.config, log_path=log_path)
    try:
        _run(args)
    except SystemExit:
        raise
    except Exception:           # noqa: BLE001 — нужен диагноз, не стектрейс в никуда
        runlog.set_stage("аварийное завершение")
        log.critical("=== ИСКЛЮЧЕНИЕ в private_realty_crawl ===", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
