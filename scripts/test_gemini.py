#!/usr/bin/env python3
"""Проверка движка Gemini на реальном кропе из базы.
Ключ читается из GEMINI_API_KEY, в файл не попадает."""
import os, sqlite3, sys

# Скрипт лежит в scripts/, а пакет banner_parser — в корне проекта.
# Python кладёт в sys.path папку СКРИПТА, а не текущий каталог, поэтому корень
# добавляем явно: иначе ModuleNotFoundError даже при запуске из корня.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

key = os.getenv("GEMINI_API_KEY")
if not key:
    sys.exit("GEMINI_API_KEY не задан. Сначала: export GEMINI_API_KEY=...")
print(f"ключ виден, длина {len(key)}, начинается на {key[:4]}...")
if not key.startswith("AIza"):
    print("ВНИМАНИЕ: ключи Gemini API обычно начинаются на 'AIza'.")
    print("Если дальше будет 401/403 — скорее всего скопирован не тот токен.")

from banner_parser.config import Config
from banner_parser.ocr.engines import GeminiBackend

cfg = Config.load(None)
model = cfg.get("ocr.gemini_model", "gemini-2.5-flash")
print(f"модель: {model}\n")

# берём кроп, на котором easyocr выдал мусор — есть с чем сравнить
c = sqlite3.connect(cfg.get("storage.db_path", "data/banners.sqlite"))
rows = c.execute("""select crop_image_path, ifnull(text,''), ifnull(ocr_engine,'')
                    from banners where crop_image_path is not null
                    order by rowid desc limit 40""").fetchall()

be = GeminiBackend(model=model, max_side=cfg.get("ocr.vlm_max_side", 1024))
done = 0
for path, old, eng in rows:
    if not path or not os.path.exists(path):
        continue
    from PIL import Image
    img = Image.open(path)
    print(f"--- {os.path.basename(path)}  {img.size[0]}x{img.size[1]}")
    print(f"    было ({eng}): {old[:70]!r}")
    r = be.read(img)
    print(f"    стало (gemini): {(r.text or '')[:70]!r}")
    print(f"    рекламодатель={r.advertiser!r} категория={r.category!r}")
    from banner_parser.ocr import extract_contacts
    ct = extract_contacts(r.text or "")
    print(f"    ТЕЛЕФОНЫ: {ct.phones}  сайты: {ct.sites}")
    done += 1
    if done >= 3:
        break
print(f"\nпроверено кропов: {done}")
