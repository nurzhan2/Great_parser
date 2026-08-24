import os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from banner_parser.config import Config
from banner_parser.ocr.engines import build_backend
from banner_parser.ocr import extract_contacts

# Сравнение движка на кропах, где прежний VLM УЖЕ прочитал текст. Мелкие кропы
# со слабых срабатываний тут бесполезны: на них и Sonnet возвращал пустоту,
# и сравнивать нечего.
cfg = Config.load(None)
be = build_backend(cfg)
print("движок:", be.name, "| модель:", getattr(be, "model", "-"), "\n")

c = sqlite3.connect(cfg.get("storage.db_path", "data/banners.sqlite"))
rows = c.execute("""select crop_image_path, ifnull(text,'') from banners
                    where ocr_engine like 'vlm%' and trim(ifnull(text,''))<>''
                    and crop_image_path is not null
                    order by rowid desc limit 30""").fetchall()

done = 0
for path, old in rows:
    if not path or not os.path.exists(path):
        continue
    img = Image.open(path)
    if min(img.size) < 300:
        continue
    print(f"--- {os.path.basename(path)}  {img.size[0]}x{img.size[1]}")
    print(f"    было (Sonnet 4.6): {old[:80]!r}")
    r = be.read(img)
    print(f"    стало            : {(r.text or '')[:80]!r}")
    print(f"    рекламодатель={r.advertiser!r}  категория={r.category!r}")
    print(f"    ТЕЛЕФОНЫ: {extract_contacts(r.text or '').phones}\n")
    done += 1
    if done >= 4:
        break
print(f"сравнено кропов: {done}")
if not done:
    print("подходящих кропов не нашлось — нужен прогон с новыми настройками")
