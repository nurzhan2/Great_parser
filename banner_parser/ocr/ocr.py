"""OCR баннера (easyocr, лениво) и извлечение телефонов из текста."""
from __future__ import annotations

import logging
import re
from typing import Optional

from PIL import Image

log = logging.getLogger(__name__)


class OcrEngine:
    """Обёртка над easyocr. Если пакет не установлен — read() вернёт ''."""

    def __init__(self, languages: Optional[list[str]] = None, gpu: bool = True):
        self.languages = languages or ["ru", "en"]
        self.gpu = gpu
        self._reader = None
        self._unavailable = False

    def _load(self):
        if self._reader is None and not self._unavailable:
            try:
                import easyocr  # lazy
                self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            except Exception as e:  # noqa: BLE001
                log.warning("easyocr недоступен (%s) — OCR-этап пропускается", e)
                self._unavailable = True
        return self._reader

    def read(self, image: Image.Image) -> str:
        import numpy as np
        reader = self._load()
        if reader is None:
            return ""
        try:
            lines = reader.readtext(np.asarray(image), detail=0, paragraph=True)
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            log.warning("OCR error: %s", e)
            return ""


# Российские телефоны: +7 / 8 / 7, затем 10 цифр с любыми разделителями.
_PHONE_RE = re.compile(
    r"(?:\+7|8|7)[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{2}[\s\-\(\)]*\d{2}")


def extract_phones(text: str) -> list[str]:
    """Находит и нормализует телефоны в формат +7XXXXXXXXXX (без дублей)."""
    found: list[str] = []
    for m in _PHONE_RE.finditer(text or ""):
        digits = re.sub(r"\D", "", m.group())
        if len(digits) == 11 and digits[0] in ("7", "8"):
            norm = "+7" + digits[1:]
            if norm not in found:
                found.append(norm)
    return found


def build_ocr(cfg) -> Optional[OcrEngine]:
    if not cfg.get("ocr.enabled", True):
        return None
    return OcrEngine(
        languages=cfg.get("ocr.languages", ["ru", "en"]),
        gpu=cfg.get("ocr.gpu", True),
    )
