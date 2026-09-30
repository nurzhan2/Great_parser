import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.basicConfig(level=logging.ERROR)
import numpy as np
from PIL import Image, ImageOps
import easyocr
from banner_parser.pipeline import _has_realty_word, _looks_like_phone

P = "data/images/1300727172_674714083_23_1780483680_0.jpg"
img = Image.open(P).convert("RGB")
rd = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
DIG = "0123456789+-() "

def run(label, im, digits=False):
    t0 = time.monotonic()
    txt = " ".join(rd.readtext(np.array(im), detail=0, **({"allowlist": DIG} if digits else {})))
    dt = time.monotonic() - t0
    w, p = _has_realty_word(txt), _looks_like_phone(txt)
    print(f"{label:28} {dt:4.1f}с  слово={'да' if w else '—'} номер={'да' if p else '—'}  {txt[:60]!r}")

x2 = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
g = x2.convert("L")
inv = ImageOps.invert(g)                       # белое на красном -> тёмное на светлом
r, gch, b = x2.split()
# Белый текст ярок во ВСЕХ каналах, красный фон ярок только в R.
# Берём минимум каналов: фон гаснет, текст остаётся.
mn = Image.fromarray(np.minimum(np.minimum(np.array(r), np.array(gch)), np.array(b)))
mn_inv = ImageOps.invert(mn)

run("x2 серое", g)
run("x2 инверсия", inv)
run("x2 min-канал", mn)
run("x2 min-канал инверсия", mn_inv)
run("x2 min-канал инв, цифры", mn_inv, digits=True)
run("x2 инверсия, цифры", inv, digits=True)
