"""Регрессия: находит ли новый набор промптов эталонную табличку
«ПРОДАЮ 8-985-970-19-42» и читается ли номер. В базу не пишет."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)

from banner_parser.config import Config
from banner_parser.detect import OwlDetector
from banner_parser.ocr import extract_contacts
from banner_parser.ocr.engines import build_backend
from banner_parser.pipeline import Pipeline
from banner_parser.yandex import Panorama

LON, LAT = 38.051384, 55.612705
cfg = Config.load(None)

NEW_P = [
    "a small red sign with white text",
    "a red rectangular placard with a phone number",
    "a small rectangular sign with printed text",
    "a handwritten paper notice",
    "a small placard with a telephone number",
    "a printed announcement sheet",
    "a real estate sign with a phone number",
    "a house for sale or rent sign",
    "a land for sale sign",
    "a property advertisement by owner",
]
NEW_N = list(cfg.get("detector.negative_prompts", [])) + [
    "a wooden fence", "a corrugated metal fence", "a brick wall",
    "a metal gate", "a concrete wall",
]

det = OwlDetector(model_name=cfg.get("detector.owl_model"),
                  conf=cfg.get("detector.conf", 0.30),
                  prompts=NEW_P, negative_prompts=NEW_N,
                  pos_strong=cfg.get("detector.pos_strong", 0.25),
                  pos_weak=cfg.get("detector.pos_weak", 0.10),
                  neg_margin=cfg.get("detector.neg_margin", 0.10))

pipe = Pipeline(cfg)
ocr = build_backend(cfg)
try:
    ref = pipe.meta.parse(pipe.meta.by_coords(LON, LAT))
    pano = Panorama(ref, pipe.http, pipe.workers)
    img = pano.stitch(cfg.get("panorama.overview_zoom", 2))
    dets = sorted(det.detect(img), key=lambda d: d.score, reverse=True)
    print(f"адрес: {ref.address}, кандидатов: {len(dets)}")
    print("читаю 8 самых уверенных:\n")
    found = False
    for i, d in enumerate(dets[:8]):
        crop = pano.crop_detection(d, cfg.get("panorama.crop_zoom", 0),
                                   pad=cfg.get("detector.crop_pad", 0.20))
        if crop is None or min(crop.size) < 40:
            continue
        r = ocr.read(crop)
        txt = (r.text or "").strip()
        ph = extract_contacts(txt).phones
        mark = "   <<< ЭТАЛОН" if "9859701942" in str(ph).replace("+7", "7") else ""
        print(f"  #{i} score={d.score:.2f} {crop.size[0]}x{crop.size[1]} "
              f"тел={ph}{mark}")
        if txt:
            print(f"      {txt[:70]!r}")
        if mark:
            found = True
    print()
    print("*** РЕГРЕССИЯ ПРОЙДЕНА: эталон найден и прочитан ***" if found
          else "!!! ЭТАЛОН ПОТЕРЯН — новые промпты его не находят")
finally:
    pipe.close()
