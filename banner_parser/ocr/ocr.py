"""Фасад OCR: выбор движка из конфига плюс кэш результатов по хэшу картинки."""
from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from .engines import EasyOcrBackend, OcrResult, ResultCache, build_backend

log = logging.getLogger(__name__)


class OcrEngine:
    """Обёртка над выбранным движком. Кэширует результаты по хэшу кропа —
    повторный прогон тех же картинок не стоит ни времени, ни денег."""

    def __init__(self, backend, cache_path: Optional[str] = None, fallback=None):
        self.backend = backend
        # Запасной движок нужен, когда основной отказал совсем (кончился баланс
        # API, нет сети). 31% распознанных слов лучше нуля: бренд иногда
        # читается, и запись перестаёт быть пустой.
        self.fallback = fallback
        self.cache = ResultCache(cache_path) if cache_path else None

    def recognize(self, image: Image.Image) -> OcrResult:
        if self.cache is not None:
            key = ResultCache.key(image, self.backend.name)
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        res = self.backend.read(image)
        if res.failed and self.fallback is not None:
            log.info("движок %s не смог — пробуем запасной %s",
                     self.backend.name, self.fallback.name)
            alt = self.fallback.read(image)
            if not alt.failed:
                res = alt
        # Неудачу не кэшируем: движок мог отказать временно (сеть, лимит API).
        if self.cache is not None and not res.failed:
            self.cache.put(ResultCache.key(image, res.engine), res)
        return res

    def read(self, image: Image.Image) -> str:
        """Совместимость со старым интерфейсом — только текст."""
        return self.recognize(image).text

    def stats(self) -> str:
        if self.cache is None:
            return "кэш выключен"
        return f"кэш: попаданий {self.cache.hits}, промахов {self.cache.misses}"

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()


def build_ocr(cfg) -> Optional[OcrEngine]:
    if not cfg.get("ocr.enabled", True):
        return None
    backend = build_backend(cfg)
    fb = None
    name = (cfg.get("ocr.backend", "easyocr") or "easyocr").lower()
    if name in ("vlm", "openai") and cfg.get("ocr.fallback_easyocr", True):
        fb = EasyOcrBackend(languages=cfg.get("ocr.languages", ["ru", "en"]),
                            gpu=cfg.get("ocr.gpu", False))
    log.info("OCR-движок: %s%s", backend.name,
             f" (запасной: {fb.name})" if fb else "")
    return OcrEngine(backend, cache_path=cfg.get("ocr.cache_path", "data/ocr_cache.sqlite"),
                     fallback=fb)
