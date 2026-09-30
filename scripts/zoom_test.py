"""Сколько выигрываем на zoom и теряем ли эталон.

Детекция — 59% времени обхода, и она квадратично зависит от площади
картинки. overview_zoom: 2 -> 3 уменьшает картинку вчетверо. Вопрос
один: переживёт ли это эталонная табличка «ПРОДАЮ 8-985-970-19-42».
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)

from banner_parser.config import Config
from banner_parser.detect import OwlDetector
from banner_parser.ocr import extract_contacts
from banner_parser.ocr.engines import build_backend
from banner_parser.pipeline import Pipeline
from banner_parser.yandex import Panorama

LON, LAT = 38.051384, 55.612705      # эталон
cfg = Config.load(None)
det = OwlDetector(model_name=cfg.get("detector.owl_model"),
                  conf=cfg.get("detector.conf", 0.30),
                  prompts=cfg.get("detector.prompts"),
                  negative_prompts=cfg.get("detector.negative_prompts"),
                  pos_strong=cfg.get("detector.pos_strong", 0.25),
                  pos_weak=cfg.get("detector.pos_weak", 0.28),
                  neg_margin=cfg.get("detector.neg_margin", 0.10))

pipe = Pipeline(cfg)
ocr = build_backend(cfg)
try:
    ref = pipe.meta.parse(pipe.meta.by_coords(LON, LAT))
    for zoom in (2, 3):
        pano = Panorama(ref, pipe.http, pipe.workers)
        t0 = time.monotonic()
        img = pano.stitch(zoom)
        t_st = time.monotonic() - t0
        t0 = time.monotonic()
        dets = sorted(det.detect(img), key=lambda d: d.score, reverse=True)
        t_det = time.monotonic() - t0
        print(f"\n=== zoom {zoom}: картинка {img.size[0]}x{img.size[1]}, "
              f"сшивка {t_st:.1f} с, ДЕТЕКЦИЯ {t_det:.1f} с, кандидатов {len(dets)}")
        found = False
        for i, d in enumerate(dets[:6]):
            crop = pano.crop_detection(d, cfg.get("panorama.crop_zoom", 0),
                                       pad=cfg.get("detector.crop_pad", 0.20))
            if crop is None or min(crop.size) < 40:
                continue
            r = ocr.read(crop)
            ph = extract_contacts(r.text or "").phones
            mark = "  <<< ЭТАЛОН" if "9859701942" in str(ph) else ""
            print(f"   #{i} score={d.score:.2f} тел={ph}{mark}")
            if mark:
                found = True
        print(f"   ЭТАЛОН: {'НАЙДЕН' if found else 'ПОТЕРЯН'}")
finally:
    pipe.close()
