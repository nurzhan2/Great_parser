"""Seed-точки для обхода: простая сетка или точки вдоль дорожного графа OSM.

Для «наружки» точки нужно сажать на дороги (billboard'ы стоят вдоль трасс),
а не во дворы. road_seeds() берёт дорожную сеть из OpenStreetMap через osmnx;
grid_seeds() — грубый fallback без внешних зависимостей.
"""
from __future__ import annotations

import logging
import math
from typing import Iterator

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


def grid_seeds(bbox: BBox, step_m: float = 150.0) -> Iterator[tuple[float, float]]:
    """Равномерная сетка точек внутри bbox с шагом ~step_m метров."""
    min_lon, min_lat, max_lon, max_lat = bbox
    dlat = step_m / 111_320.0
    mid_lat = (min_lat + max_lat) / 2
    dlon = step_m / (111_320.0 * math.cos(math.radians(mid_lat)))
    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            yield lon, lat
            lon += dlon
        lat += dlat


# Классы дорог, вдоль которых вообще стоит наружная реклама. network_type="drive"
# тянет ещё и service/residential, то есть внутридворовые проезды — обход по ним
# уходит во дворы, где рекламы нет физически.
ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary")


def _graph_from_bbox(ox, bbox: BBox, road_classes, network_type: str):
    """Совместимость osmnx 1.x/2.x: в 2.0 сигнатура graph_from_bbox сменилась
    с четырёх позиционных (north, south, east, west) на один кортеж
    bbox=(left, bottom, right, top)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    custom = None
    if road_classes:
        custom = '["highway"~"' + "|".join(road_classes) + '"]'
    try:                                    # osmnx >= 2.0
        if custom:
            return ox.graph_from_bbox(bbox=(min_lon, min_lat, max_lon, max_lat),
                                      custom_filter=custom)
        return ox.graph_from_bbox(bbox=(min_lon, min_lat, max_lon, max_lat),
                                  network_type=network_type)
    except TypeError:                       # osmnx 1.x
        log.info("osmnx со старой сигнатурой graph_from_bbox — используем её")
        if custom:
            return ox.graph_from_bbox(max_lat, min_lat, max_lon, min_lon,
                                      custom_filter=custom)
        return ox.graph_from_bbox(max_lat, min_lat, max_lon, min_lon,
                                  network_type=network_type)


def road_seeds(bbox: BBox, step_m: float = 40.0, network_type: str = "drive",
               road_classes=ROAD_CLASSES) -> Iterator[tuple[float, float]]:
    """Точки вдоль дорог OSM с шагом step_m. Требует osmnx (+ shapely, networkx).

    road_classes=None вернёт прежнее поведение (все проезжие дороги, включая
    дворовые); по умолчанию оставляем только магистрали и улицы.
    """
    try:
        import osmnx as ox
    except Exception as e:  # noqa: BLE001
        log.warning("osmnx недоступен (%s) — откат к grid_seeds", e)
        yield from grid_seeds(bbox, step_m=150.0)
        return

    try:
        G = _graph_from_bbox(ox, bbox, road_classes, network_type)
    except Exception as e:  # noqa: BLE001
        # Пустой ответ Overpass на узкий фильтр — обычное дело для маленькой рамки.
        log.warning("не удалось построить дорожный граф (%s: %s) — откат к сетке",
                    type(e).__name__, e)
        yield from grid_seeds(bbox, step_m=150.0)
        return
    log.info("дорожный граф: %d рёбер, классы %s",
             len(G.edges), ",".join(road_classes) if road_classes else network_type)
    G = ox.project_graph(G)
    _, edges = ox.graph_to_gdfs(G)
    edges_ll = edges.to_crs(epsg=4326)
    seen: set[tuple[float, float]] = set()
    for geom in edges_ll.geometry:
        length_deg_ok = geom.length
        if length_deg_ok == 0:
            continue
        n = max(1, int(_geo_len_m(geom) / step_m))
        for k in range(n + 1):
            p = geom.interpolate(k / n, normalized=True)
            key = (round(p.x, 5), round(p.y, 5))
            if key not in seen:
                seen.add(key)
                yield p.x, p.y


def _geo_len_m(geom) -> float:
    """Приблизительная длина линии (град→метры) по средней широте."""
    coords = list(geom.coords)
    total = 0.0
    for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
        mlat = math.radians((y0 + y1) / 2)
        dx = (x1 - x0) * 111_320.0 * math.cos(mlat)
        dy = (y1 - y0) * 111_320.0
        total += math.hypot(dx, dy)
    return total
