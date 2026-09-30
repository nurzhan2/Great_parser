"""Можно ли выжать из бесплатного easyocr телефон надёжно.

Три приёма, каждый бесплатный:
  1. allowlist — ограничить алфавит цифрами и разделителями. Ошибка
     '05-05' -> '05-15' случилась при выборе среди всех символов;
     на голых цифрах путать не с чем.
  2. апскейл — мелкий текст читается лучше после увеличения.
  3. контраст — выцветшая краска на заборе.
Эталон — телефоны, которые DeepSeek уже вытащил верно.
"""
import os, re, sqlite3, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import easyocr

from banner_parser.config import Config
from banner_parser.ocr import extract_contacts

cfg = Config.load(None)
c = sqlite3.connect(cfg.get("storage.db_path", "data/banners.sqlite"))
rows = c.execute("""select crop_image_path, phones from banners
                    where phones is not null and phones<>'' and phones<>'[]'
                    and crop_image_path is not null order by rowid desc limit 20""").fetchall()
samples = [(p, ph) for p, ph in rows if p and os.path.exists(p)][:6]
print(f"кропов: {len(samples)}\n")

reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
DIGITS = "0123456789+-() "

def variants(img):
    yield "как есть        ", img
    w, h = img.size
    yield "апскейл x3      ", img.resize((w * 3, h * 3), Image.LANCZOS)
    g = ImageOps.autocontrast(img.convert("L"))
    g = ImageEnhance.Sharpness(g).enhance(2.0)
    yield "контраст+апскейл", g.resize((w * 3, h * 3), Image.LANCZOS)

def phones_from(txt):
    return extract_contacts(txt).phones

ok = {}
for path, ref in samples:
    img = Image.open(path).convert("RGB")
    print(f"--- {os.path.basename(path)} {img.size[0]}x{img.size[1]}  эталон={ref}")
    for label, im in variants(img):
        arr = np.array(im)
        for mode, kw in (("весь алфавит", {}), ("только цифры", {"allowlist": DIGITS})):
            t0 = time.monotonic()
            try:
                parts = reader.readtext(arr, detail=0, **kw)
            except Exception as e:
                print(f"    {label} | {mode}: ОШИБКА {type(e).__name__}")
                continue
            txt = " ".join(parts)
            ph = phones_from(txt)
            dt = time.monotonic() - t0
            hit = ph and all(p in ref for p in ph)
            key = f"{label}|{mode}"
            ok[key] = ok.get(key, 0) + (1 if hit else 0)
            mark = "  ВЕРНО" if hit else ("  НЕВЕРНО" if ph else "")
            print(f"    {label} | {mode}: {dt:4.1f}с {str(ph)[:40]:42}{mark}")
    print()

print("=== ИТОГ (сколько телефонов угадано верно из", len(samples), ")")
for k, v in sorted(ok.items(), key=lambda x: -x[1]):
    print(f"   {v}/{len(samples)}  {k}")
