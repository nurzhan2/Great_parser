"""Нормализация рекламодателя и обогащение контактов по справочнику брендов.

Две отдельные задачи, обе завязаны на brands.yaml:

  1. Нормализация. VLM возвращает то, что написано на щите: «Wildberries»,
     «WB», «вайлдберриз». Без приведения к каноничному имени база не
     группируется — замер показал, что «День Индии» и «India Day» с соседних
     щитов считались разными рекламодателями. Неизвестный бренд НЕ
     выбрасывается: сохраняется как есть с пометкой brand_matched=False.

  2. Обогащение. Контакты со щита читаются надёжно только при крупном шрифте
     (замер: 1 верный сайт из 4 на мелком). Контакты из справочника — данные
     другой природы и другой надёжности, поэтому они возвращаются отдельными
     полями и в базе лежат в отдельных колонках. Смешивать их с прочитанным
     со щита нельзя: продавец должен видеть, откуда взялся номер.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "brands.yaml"


@dataclass
class BrandInfo:
    canonical: Optional[str] = None      # каноничное имя застройщика/агентства
    matched: bool = False                # нашёлся ли в справочнике
    role: Optional[str] = None           # застройщик | агентство | банк
    complex_name: Optional[str] = None   # ЖК, если опознан по названию
    site: Optional[str] = None           # контакты ИЗ СПРАВОЧНИКА, не со щита
    phone: Optional[str] = None
    category: Optional[str] = None


def _norm(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", " ", s).strip()


class BrandDirectory:
    def __init__(self, path: Optional[str] = None):
        self.by_alias: dict[str, dict] = {}
        self.by_complex: dict[str, tuple] = {}
        p = Path(path) if path else _DEFAULT_PATH
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except OSError as e:
            log.warning("справочник брендов недоступен (%s) — нормализация выключена", e)
            return
        for canonical, info in (data.get("brands") or {}).items():
            info = info or {}
            entry = {"canonical": canonical, "site": info.get("site"),
                     "phone": info.get("phone"),
                     "category": info.get("category") or "недвижимость",
                     "role": info.get("role")}
            for k in [canonical] + list(info.get("aliases") or []):
                nk = _norm(str(k))
                if nk:
                    self.by_alias[nk] = entry
            # ЖК резолвится в застройщика, но остаётся ОТДЕЛЬНОЙ сущностью:
            # запоминаем, каким именно комплексом было совпадение.
            for c in (info.get("complexes") or []):
                nc = _norm(str(c))
                if nc:
                    self.by_complex[nc] = (entry, str(c))
        log.info("справочник: %d рекламодателей, %d алиасов, %d ЖК",
                 len(data.get("brands") or {}), len(self.by_alias), len(self.by_complex))

    def lookup(self, raw: Optional[str], complex_hint: Optional[str] = None) -> BrandInfo:
        """Найти рекламодателя по имени со щита и/или по названию ЖК."""
        # Сначала пробуем ЖК: «Квартал Домашний» на щите может стоять без
        # упоминания застройщика, но однозначно указывает на него.
        for src in (complex_hint, raw):
            nc = _norm(src or "")
            if not nc:
                continue
            hit = self.by_complex.get(nc)
            if hit is None:
                for cname in sorted(self.by_complex, key=len, reverse=True):
                    if len(cname) >= 5 and cname in nc:
                        hit = self.by_complex[cname]
                        break
            if hit is not None:
                e, cname = hit
                return BrandInfo(canonical=e["canonical"], matched=True,
                                 role=e.get("role"), complex_name=cname,
                                 site=e.get("site"), phone=e.get("phone"),
                                 category=e.get("category"))
        if not raw:
            return BrandInfo()
        n = _norm(raw)
        if not n:
            return BrandInfo()

        entry = self.by_alias.get(n)
        if entry is None:
            # Щит часто содержит бренд внутри фразы: «Квартал Домашний от
            # Самолёта», «до офиса продаж КИНОМАКС». Ищем вхождение алиаса
            # словом, начиная с самых длинных — иначе короткий алиас вроде
            # «вб» совпадёт внутри случайного слова.
            for alias in sorted(self.by_alias, key=len, reverse=True):
                if len(alias) < 3:
                    continue
                if re.search(r"(?:^|\s)" + re.escape(alias) + r"(?:\s|$)", n):
                    entry = self.by_alias[alias]
                    break
        if entry is None:
            # Не нашли — не выбрасываем: сохраняем как есть, но помечаем.
            return BrandInfo(canonical=raw.strip(), matched=False)
        return BrandInfo(canonical=entry["canonical"], matched=True,
                         role=entry.get("role"),
                         site=entry.get("site"), phone=entry.get("phone"),
                         category=entry.get("category"))


def build_brand_directory(cfg) -> Optional[BrandDirectory]:
    if not cfg.get("brands.enabled", True):
        return None
    return BrandDirectory(cfg.get("brands.path", None))
