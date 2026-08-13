"""OCR баннера через easyocr (ленивая загрузка). Разбор контактов — в contacts.py."""
from __future__ import annotations

import logging
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


def build_ocr(cfg) -> Optional[OcrEngine]:
    if not cfg.get("ocr.enabled", True):
        return None
    return OcrEngine(
        languages=cfg.get("ocr.languages", ["ru", "en"]),
        gpu=cfg.get("ocr.gpu", True),
    )
