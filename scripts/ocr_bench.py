"""Сравнение движков OCR на реальных кропах: качество, время, цена."""
import os, sqlite3, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)
from PIL import Image
from banner_parser.config import Config
from banner_parser.ocr import extract_contacts

cfg = Config.load(None)
c = sqlite3.connect(cfg.get("storage.db_path", "data/banners.sqlite"))

# Кропы с ИЗВЕСТНЫМ ответом: там, где DeepSeek уже вытащил телефон.
rows = c.execute("""select crop_image_path, text, phones from banners
                    where phones is not null and phones<>'' and phones<>'[]'
                    and crop_image_path is not null order by rowid desc limit 12""").fetchall()
samples = [(p, t, ph) for p, t, ph in rows if p and os.path.exists(p)][:5]
print(f"кропов для теста: {len(samples)}\n")

from banner_parser.ocr.engines import EasyOcrBackend, PaddleOcrBackend
engines = []
try:
    engines.append(("easyocr", EasyOcrBackend(languages=["ru", "en"], gpu=False)))
except Exception as e:
    print("easyocr не поднялся:", e)
try:
    engines.append(("paddle ", PaddleOcrBackend(lang=cfg.get("ocr.paddle_lang", "ru"))))
except Exception as e:
    print("paddle не поднялся:", type(e).__name__, str(e)[:120])

for path, ref_text, ref_ph in samples:
    img = Image.open(path)
    print(f"--- {os.path.basename(path)}  {img.size[0]}x{img.size[1]}")
    print(f"    ЭТАЛОН (deepseek): {(ref_text or '')[:60]!r}  тел={ref_ph}")
    for name, be in engines:
        t0 = time.monotonic()
        try:
            r = be.read(img)
            txt = (r.text or "").strip()
            ph = extract_contacts(txt).phones
            dt = time.monotonic() - t0
            hit = "  ТЕЛЕФОН СОВПАЛ" if ph and str(ph) == str(ref_ph) else ""
            print(f"    {name}: {dt:5.1f} с  тел={ph}{hit}")
            print(f"             {txt[:60]!r}")
        except Exception as e:
            print(f"    {name}: ОШИБКА {type(e).__name__}: {str(e)[:70]}")
    print()
