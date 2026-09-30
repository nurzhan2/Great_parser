"""Правило «только слово о недвижимости» на всех эталонах + красной табличке.
Плюс замер: помогает ли увеличение x2 надёжнее ловить слово."""
import os, sqlite3, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.basicConfig(level=logging.ERROR)
import numpy as np
from PIL import Image
import easyocr
from banner_parser.pipeline import _has_realty_word, _looks_like_phone

c = sqlite3.connect("data/banners.sqlite")
rows = c.execute("""select crop_image_path, ifnull(text,''), score from banners
                    where phones is not null and phones<>'' and phones<>'[]'
                    and crop_image_path is not null order by rowid desc limit 30""").fetchall()
samples = [(p, t, s) for p, t, s in rows if p and os.path.exists(p)][:8]
red = "data/images/1300727172_674714083_23_1780483680_0.jpg"
samples.append((red, "ПРОДАЮ 8-985-970-19-42 (красная табличка)", 0.68))

# что из них НА САМОМ ДЕЛЕ недвижимость — размечено вручную по тексту DeepSeek
REALTY = {"1300988712", "1300656060", "1300727172"}

rd = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
print(f"{'кроп':14} {'на деле':10} {'x1':>10} {'x2':>10}   текст x2")
print("-" * 90)
res = {1: dict(tp=0, fp=0, fn=0), 2: dict(tp=0, fp=0, fn=0)}
for p, dtxt, sc in samples:
    key = os.path.basename(p)[:10]
    truth = key in REALTY
    img = Image.open(p).convert("RGB")
    outs = {}
    for k in (1, 2):
        im = img if k == 1 else img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        txt = " ".join(rd.readtext(np.array(im), detail=0))
        hit = _has_realty_word(txt)
        outs[k] = (hit, txt)
        if hit and truth: res[k]["tp"] += 1
        elif hit and not truth: res[k]["fp"] += 1
        elif not hit and truth: res[k]["fn"] += 1
    print(f"{key:14} {'НЕДВИЖ' if truth else 'мусор':10} "
          f"{'ПИШЕМ' if outs[1][0] else '—':>10} {'ПИШЕМ' if outs[2][0] else '—':>10}   {outs[2][1][:40]!r}")

print("\n=== ИТОГ (3 объявления о недвижимости, 6 мусора)")
for k in (1, 2):
    r = res[k]
    print(f"  x{k}: найдено {r['tp']}/3 недвижимости, пропущено {r['fn']}, "
          f"ложных {r['fp']}/6")
