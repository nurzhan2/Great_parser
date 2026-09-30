"""Пересборка вкладки «тест»: только записи, проходящие строгий фильтр.

Перед очисткой всё содержимое вкладки сохраняется в CSV — удалённое
можно вернуть. Ошибка, которую исправляем: sync_sheets.py залил все
488 записей базы, включая мусор из прогонов до введения фильтра.
"""
import csv, datetime, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)

from banner_parser.config import Config
from banner_parser.export.google_sheets import build_google_sheets
from banner_parser.pipeline import _is_target_lead

cfg = Config.load(None)
g = build_google_sheets(cfg)
if g is None:
    sys.exit("таблица не подключена")
svc = g._svc if hasattr(g, "_svc") else g.service
sid, tab = g.spreadsheet_id, g.sheet_name
vals = svc.spreadsheets().values()

# 1. резервная копия
cur = vals.get(spreadsheetId=sid, range=f"'{tab}'").execute().get("values", [])
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = f"data/sheet_backup_{tab}_{stamp}.csv"
with open(bak, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(cur)
print(f"резерв: {bak}, строк {len(cur)}")
# Источник строк: самая полная резервная копия, а не текущая вкладка.
# Иначе запись, удалённая прошлой пересборкой, не вернётся, даже если
# фильтр её теперь пропускает.
import glob
full = max(glob.glob(f"data/sheet_backup_{tab}_*.csv"),
           key=lambda p: sum(1 for _ in open(p, encoding="utf-8")))
with open(full, encoding="utf-8") as f:
    cur = list(csv.reader(f))
print(f"источник строк: {full}, строк {len(cur)}")
header = cur[0] if cur else None

# 2. отбор записей по действующему фильтру
mode = cfg.get("filter.mode", "realty")
req_phone = bool(cfg.get("filter.require_phone", True))


class R:
    pass


c = sqlite3.connect(cfg.get("storage.db_path", "data/banners.sqlite"))
c.row_factory = sqlite3.Row
keep_ids = set()
for row in c.execute("select * from banners"):
    r = R()
    for k in row.keys():
        setattr(r, k, row[k])
    if _is_target_lead(r, req_phone, mode):
        keep_ids.add(row["banner_id"])
print(f"в базе всего: {c.execute('select count(*) from banners').fetchone()[0]}, "
      f"проходят фильтр ({mode}): {len(keep_ids)}")

# 3. оставляем в таблице только их
id_col = 0
kept = [header] + [r for r in cur[1:] if r and r[id_col] in keep_ids] if header else []
vals.clear(spreadsheetId=sid, range=f"'{tab}'").execute()
if kept:
    vals.update(spreadsheetId=sid, range=f"'{tab}'!A1",
                valueInputOption="RAW", body={"values": kept}).execute()
print(f"в таблице теперь: {len(kept) - 1} записей (было {len(cur) - 1})")
for r in kept[1:]:
    print("   ", " | ".join((r + [''] * 20)[i][:34] for i in (2, 13)))
