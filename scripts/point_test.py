import logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
from banner_parser.config import Config
from banner_parser.pipeline import Pipeline

LON, LAT = 38.051384, 55.612705   # забор с «ПРОДАЮ 8-985-970-19-42»

cfg = Config.load(None)
p = Pipeline(cfg)
print(f'=== обрабатываю точку {LON}, {LAT}')
try:
    recs = p.process_point(LON, LAT)
    print(f'\n=== ПОЛУЧЕНО ЗАПИСЕЙ: {len(recs)}')
    for r in recs:
        print(f'  [{r.category}] score={r.score:.2f} тел={r.phones} азимут={r.bearing_deg}')
        print(f'      текст: {(r.text or "")[:110]!r}')
        print(f'      кроп : {r.crop_image_path}')
    found = [r for r in recs if 'ПРОДА' in (r.text or '').upper()
             or '9709701942' in str(r.phones).replace('-', '').replace('+7', '')
             or '970' in str(r.phones)]
    print()
    print('*** ЭТАЛОН НАЙДЕН ***' if found else '!!! эталон НЕ найден среди сохранённых')
finally:
    p.close()
