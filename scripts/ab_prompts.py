"""A/B промптов детектора на одних и тех же панорамах.

В базу ничего не пишет и OCR не тратит: сравнивается только то, сколько
кандидатов даёт каждый набор и какого они размера. Забор от таблички
отличаем по доле кадра — забор занимает заметную часть панорамы,
табличка это крошечный прямоугольник.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)

from banner_parser.config import Config
from banner_parser.detect import OwlDetector
from banner_parser.pipeline import Pipeline
from banner_parser.yandex import Panorama

# Первая точка — эталон «ПРОДАЮ 8-985-970-19-42»: регрессионная проверка,
# новый набор промптов обязан его сохранить.
POINTS = [
    (38.051384, 55.612705),
    (38.0470, 55.6150),
    (38.0560, 55.6100),
]

cfg = Config.load(None)
OLD_P = cfg.get("detector.prompts", [])
OLD_N = cfg.get("detector.negative_prompts", [])

# Новый набор описывает САМ ОБЪЕКТ и его ВНЕШНИЙ ВИД, а не то, к чему он
# прикреплён. Формулировки «attached to a fence» заставляли OWLv2 находить
# забор. Красный прямоугольник с белым текстом — самый частый вид таких
# табличек, и для zero-shot детектора цвет с формой сигнал куда сильнее,
# чем смысловое описание.
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
# Забор в негативах. Риск известен и задокументирован в конфиге: забор —
# фон объявления. Но при neg_margin=0.10 негатив должен ПРЕВЗОЙТИ позитив
# на 0.10, чтобы отбросить. На тесной рамке вокруг красной таблички этого
# не случится, на рамке вокруг секции профнастила — случится.
NEW_N = list(OLD_N) + [
    "a wooden fence", "a corrugated metal fence", "a brick wall",
    "a metal gate", "a concrete wall",
]


def build(p, n):
    return OwlDetector(
        model_name=cfg.get("detector.owl_model"),
        conf=cfg.get("detector.conf", 0.30),
        prompts=p, negative_prompts=n,
        pos_strong=cfg.get("detector.pos_strong", 0.25),
        pos_weak=cfg.get("detector.pos_weak", 0.10),
        neg_margin=cfg.get("detector.neg_margin", 0.10))


def summarize(dets, label):
    if not dets:
        print(f"    {label}: 0 кандидатов")
        return []
    # доля кадра, занимаемая рамкой
    areas = [((d.fx1 - d.fx0) * (d.fy1 - d.fy0), d) for d in dets]
    small = [a for a, _ in areas if a < 0.002]     # табличка
    big = [a for a, _ in areas if a >= 0.01]       # забор/стена
    print(f"    {label}: всего {len(dets)}, мелких(<0.2% кадра) {len(small)}, "
          f"крупных(>1%) {len(big)}, score max={max(d.score for d in dets):.2f}")
    return dets


pipe = Pipeline(cfg)
try:
    det_old, det_new = build(OLD_P, OLD_N), build(NEW_P, NEW_N)
    for lon, lat in POINTS:
        raw = pipe.meta.by_coords(lon, lat)
        if raw is None:
            print(f"\n{lon},{lat}: панорамы нет")
            continue
        ref = pipe.meta.parse(raw)
        pano = Panorama(ref, pipe.http, pipe.workers)
        img = pano.stitch(cfg.get("panorama.overview_zoom", 2))
        print(f"\n=== {ref.address or '(без адреса)'}  ({lon},{lat})")
        summarize(det_old.detect(img), "СТАРЫЕ")
        summarize(det_new.detect(img), "НОВЫЕ ")
finally:
    pipe.close()
