"""Что easyocr теряет на эталонах и что из этого можно вернуть бесплатно.

Для каждого кропа считаем два признака — слово о недвижимости и номер —
в четырёх вариантах обработки. Цель: понять, какой вариант даёт больше
кропов, где выживают ОБА признака. Именно оба и нужны строгому фильтру.
"""
import os, re, sqlite3, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import easyocr

from banner_parser.pipeline import _has_realty_word, _looks_like_phone

c = sqlite3.connect("data/banners.sqlite")
rows = c.execute("""select crop_image_path, phones, text, score from banners
                    where phones is not null and phones<>'' and phones<>'[]'
                    and crop_image_path is not null order by rowid desc limit 30""").fetchall()
samples = [(p, ph, t, sc) for p, ph, t, sc in rows if p and os.path.exists(p)][:8]
print(f"эталонов: {len(samples)}\n")

reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
DIG = "0123456789+-() "


def prep(img, scale, contrast):
    im = img.convert("RGB")
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    if contrast:
        g = ImageOps.autocontrast(im.convert("L"))
        im = ImageEnhance.Sharpness(g).enhance(2.0).convert("RGB")
    return im


VARIANTS = [
    ("как есть",            dict(scale=1, contrast=False), False),
    ("x2",                  dict(scale=2, contrast=False), False),
    ("x2+контраст",         dict(scale=2, contrast=True),  False),
    ("x2 + проход цифры",   dict(scale=2, contrast=False), True),
]

tally = {v[0]: dict(word=0, phone=0, both=0, sec=0.0) for v in VARIANTS}

for path, ph, dtext, sc in samples:
    img = Image.open(path)
    print(f"--- {os.path.basename(path)[:30]} {img.size[0]}x{img.size[1]} score={sc:.2f}")
    print(f"    DeepSeek читал: {(dtext or '')[:60]!r}")
    for name, kw, digits_pass in VARIANTS:
        t0 = time.monotonic()
        im = prep(img, **kw)
        txt = " ".join(reader.readtext(np.array(im), detail=0))
        if digits_pass:
            txt += " " + " ".join(reader.readtext(np.array(im), detail=0, allowlist=DIG))
        dt = time.monotonic() - t0
        w, p = _has_realty_word(txt), _looks_like_phone(txt)
        tally[name]["word"] += w; tally[name]["phone"] += p
        tally[name]["both"] += (w and p); tally[name]["sec"] += dt
        print(f"    {name:20} {dt:4.1f}с  слово={'да' if w else '—':3} номер={'да' if p else '—':3}  {txt[:55]!r}")
    print()

print("=== ИТОГ ПО 8 ЭТАЛОНАМ (сколько кропов прошло бы фильтр)")
for name, t in tally.items():
    print(f"  {name:20} слово {t['word']}/8  номер {t['phone']}/8  ОБА {t['both']}/8  "
          f"{t['sec']/len(samples):.1f} с/кроп")
