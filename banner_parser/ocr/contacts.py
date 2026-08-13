"""Извлечение контактов из распознанного текста: телефоны, сайты, telegram.

Старый парсер брал любые 11 цифр, начинающиеся с 7 или 8, и выдавал мусор
вроде +77260051860 — код 726 не бывает мобильным ни в России, ни в Казахстане,
это следы OCR, а не номер. Здесь код проверяется по диапазонам, а всё
непонятное отбрасывается или помечается ненадёжным.

Сайты и telegram на щитах встречаются чаще телефонов, поэтому вытягиваются тоже.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- телефоны --------------------------------------------------------------
# Разделители внутри номера: пробелы, скобки, дефисы, точки. OCR ещё вставляет
# случайные буквы между группами — их допускаем поштучно.
_SEP = r"[\s\.\-\(\)‐-―]*"
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?7|8)" + _SEP +
    r"(\d{3})" + _SEP + r"(\d{3})" + _SEP + r"(\d{2})" + _SEP + r"(\d{2})(?!\d)")

# Мобильные коды России: 900–999.
# Мобильные коды Казахстана: 700–708, 747, 750–751, 760–764, 771–778.
# Географические коды России: 3xx, 4xx, 8xx (Москва 495/499, Петербург 812…).
_KZ_MOBILE = ({f"70{d}" for d in range(0, 9)} | {"747"} |
              {"750", "751"} | {f"76{d}" for d in range(0, 5)} |
              {f"77{d}" for d in range(1, 9)})


def classify_code(code: str) -> str:
    """Тип номера по коду: mobile_ru | mobile_kz | geo_ru | unknown."""
    if code.startswith("9"):
        return "mobile_ru"
    if code in _KZ_MOBILE:
        return "mobile_kz"
    if code[0] in ("3", "4", "8"):
        return "geo_ru"
    return "unknown"


@dataclass
class Contacts:
    phones: list[str] = field(default_factory=list)          # надёжные
    phones_unreliable: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=list)
    telegram: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"phones": self.phones, "phones_unreliable": self.phones_unreliable,
                "sites": self.sites, "telegram": self.telegram}


def extract_phones(text: str) -> list[str]:
    """Только надёжные номера в формате +7XXXXXXXXXX (обратная совместимость)."""
    return extract_contacts(text).phones


def extract_contacts(text: str) -> Contacts:
    c = Contacts()
    raw = text or ""

    for m in _PHONE_RE.finditer(raw):
        code, a, b, d = m.groups()
        norm = f"+7{code}{a}{b}{d}"
        kind = classify_code(code)
        if kind == "unknown":
            # Не выбрасываем совсем: пусть лежит отдельно и не портит дедуп.
            if norm not in c.phones_unreliable:
                c.phones_unreliable.append(norm)
        elif norm not in c.phones:
            c.phones.append(norm)

    for m in _SITE_RE.finditer(raw):
        site = m.group(0).strip().strip(".,;:").lower()
        site = re.sub(r"^(https?://)?(www\.)?", "", site)
        if len(site) >= 5 and site not in c.sites:
            c.sites.append(site)

    for m in _TG_RE.finditer(raw):
        h = "@" + m.group(1).lower()
        if h not in c.telegram:
            c.telegram.append(h)
    return c


# --- сайты и telegram ------------------------------------------------------
# Домены верхнего уровня, которые реально встречаются на российских щитах.
_TLD = r"(?:ru|рф|com|kz|net|org|su|pro|info|shop|online|store|moscow)"
_SITE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9\-]{1,40}\." + _TLD + r"\b",
    re.IGNORECASE)
_TG_RE = re.compile(r"(?:t\.me/|telegram\.me/|@)([a-zA-Z][a-zA-Z0-9_]{3,31})")
