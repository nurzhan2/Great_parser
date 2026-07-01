"""Обход дорожного графа панорам через связи Thoroughfares (BFS)."""
from __future__ import annotations

import logging
from collections import deque
from typing import Iterator, Optional

from ..model import PanoramaRef
from .meta import MetaClient

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


def _in_bbox(ref: PanoramaRef, bbox: Optional[BBox]) -> bool:
    if bbox is None:
        return True
    return bbox[0] <= ref.lon <= bbox[2] and bbox[1] <= ref.lat <= bbox[3]


def walk_graph(meta: MetaClient, start_lon: float, start_lat: float,
               max_nodes: int = 500, bbox: Optional[BBox] = None) -> Iterator[PanoramaRef]:
    """Идём по связям панорам от стартовой точки, отдавая уникальные PanoramaRef.

    Дедуп по panoid; ограничение max_nodes и опциональная рамка bbox —
    предохранители от неконтролируемого разрастания обхода.
    """
    raw = meta.by_coords(start_lon, start_lat)
    if raw is None:
        log.warning("no panorama at start point %f,%f", start_lon, start_lat)
        return

    start = meta.parse(raw)
    seen: set[str] = {start.panoid}
    queue: deque[PanoramaRef] = deque([start])

    while queue and len(seen) <= max_nodes:
        ref = queue.popleft()
        if not _in_bbox(ref, bbox):
            continue
        yield ref
        for oid in ref.neighbor_oids:
            nraw = meta.by_oid(oid)
            if nraw is None:
                continue
            nref = meta.parse(nraw)
            if nref.panoid in seen:
                continue
            seen.add(nref.panoid)
            queue.append(nref)
    log.info("walk done: %d panoramas visited", len(seen))
