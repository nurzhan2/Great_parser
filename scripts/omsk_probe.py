import os, sys, time
sys.path.insert(0, ".")
import logging; logging.basicConfig(level=logging.WARNING)
from banner_parser.private_realty import road_seeds_private, _load_env_file
from banner_parser.config import Config
from banner_parser.pipeline import Pipeline

_load_env_file()
cfg = Config.load(None)
p = Pipeline(cfg)

# Три пробных участка Омска: юг (Ленинский), левый берег (Кировский), север (Советский)
TESTS = [("Ленинский, юг", (73.33, 54.92, 73.42, 54.97)),
         ("Кировский, левый берег", (73.24, 54.96, 73.33, 55.01)),
         ("Советский, север", (73.28, 55.03, 73.37, 55.08))]
for name, bb in TESTS:
    t0 = time.monotonic()
    pts = list(road_seeds_private(bb, 60, ("living_street", "residential", "unclassified")))
    has = 0
    for lon, lat in pts[::max(1, len(pts) // 12)][:12]:     # 12 точек вразброс
        if p.meta.by_coords(lon, lat) is not None:
            has += 1
    print(f"{name:26} seed-точек {len(pts):5}   панорамы есть в {has}/12 пробах   {time.monotonic()-t0:.0f} с")
p.close()
