"""Минимальный тест PaddleOCR-VL: одна картинка, мало потоков.
Цель — понять, инференс вообще возможен на этом сервере или его убивают."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)

IMG = sys.argv[1] if len(sys.argv) > 1 else "data/images/1300656060_674671661_23_1780483810_3.jpg"
print("картинка:", IMG, "существует:", os.path.exists(IMG), flush=True)

from paddleocr import PaddleOCRVL
print("импорт ок", flush=True)
t0 = time.monotonic()
p = PaddleOCRVL()
print(f"модель поднята за {time.monotonic()-t0:.1f} с", flush=True)

t0 = time.monotonic()
out = p.predict(IMG)
print(f"инференс за {time.monotonic()-t0:.1f} с", flush=True)

txt = []
for res in out:
    d = res if isinstance(res, dict) else (getattr(res, "json", None) or {})
    txt.append(str(d))
joined = " ".join(txt)
print("ДЛИНА ВЫВОДА:", len(joined), flush=True)
print("ВЫВОД:", joined[:600], flush=True)

sys.path.insert(0, ".")
from banner_parser.ocr import extract_contacts
print("ТЕЛЕФОНЫ:", extract_contacts(joined).phones, flush=True)
print("ожидался: +79859791299", flush=True)
