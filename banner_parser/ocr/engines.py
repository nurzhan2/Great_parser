"""Движки распознавания за одним интерфейсом: easyocr | paddle | vlm.

Зачем интерфейс: замер показал, что EasyOCR упирается в 31% слов и ни одна
его настройка не помогает — вопрос в выборе движка, а не в параметрах. Чтобы
сравнивать движки на живом обходе, а не переписывать код, выбор вынесен в
config.yaml (`ocr.backend`).

Все движки возвращают OcrResult: текст плюс необязательные поля, которые умеет
дать только VLM (рекламодатель, тема). Пайплайн берёт их, если они есть, и
откатывается к словарю ключевых слов, если нет.

Результаты кэшируются по хэшу картинки: повторный прогон тех же кропов не
должен стоить ни денег, ни времени.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class OcrResult:
    text: str = ""
    advertiser: Optional[str] = None      # кто рекламируется — умеет только VLM
    category: Optional[str] = None        # тема — умеет только VLM
    construction: Optional[str] = None    # тип конструкции — умеет только VLM
    engine: str = ""
    failed: bool = False                  # движок не смог; запись помечается
    usage: dict = field(default_factory=dict)   # токены для подсчёта стоимости


# ---- кэш ------------------------------------------------------------------
class ResultCache:
    """Кэш по SHA-256 картинки. Общий на все движки, ключ включает имя движка."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS ocr_cache (
            key TEXT PRIMARY KEY, engine TEXT, text TEXT,
            advertiser TEXT, category TEXT, construction TEXT,
            usage TEXT, created REAL)""")
        self.conn.commit()
        self.hits = self.misses = 0

    @staticmethod
    def key(image: Image.Image, engine: str) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG", compress_level=1)
        return engine + ":" + hashlib.sha256(buf.getvalue()).hexdigest()

    def get(self, k: str) -> Optional[OcrResult]:
        row = self.conn.execute(
            "SELECT engine, text, advertiser, category, construction, usage FROM ocr_cache WHERE key=?",
            (k,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return OcrResult(text=row[1] or "", advertiser=row[2], category=row[3],
                         construction=row[4], engine=row[0],
                         usage=json.loads(row[5] or "{}"))

    def put(self, k: str, r: OcrResult) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO ocr_cache VALUES (?,?,?,?,?,?,?,?)",
            (k, r.engine, r.text, r.advertiser, r.category, r.construction,
             json.dumps(r.usage), time.time()))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ---- движки ---------------------------------------------------------------
class OcrBackend:
    name = "base"

    def read(self, image: Image.Image) -> OcrResult:
        raise NotImplementedError


class EasyOcrBackend(OcrBackend):
    name = "easyocr"

    def __init__(self, languages=None, gpu: bool = False):
        self.languages = languages or ["ru", "en"]
        self.gpu = gpu
        self._reader = None
        self._dead = False

    def _load(self):
        if self._reader is None and not self._dead:
            try:
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
            except Exception as e:               # noqa: BLE001
                log.warning("easyocr недоступен (%s) — этап OCR пропускается", e)
                self._dead = True
        return self._reader

    def read(self, image: Image.Image) -> OcrResult:
        reader = self._load()
        if reader is None:
            return OcrResult(engine=self.name, failed=True)
        try:
            lines = reader.readtext(np.asarray(image), detail=0, paragraph=True)
            return OcrResult(text="\n".join(lines), engine=self.name)
        except Exception as e:                   # noqa: BLE001
            log.warning("easyocr: ошибка распознавания: %s", e)
            return OcrResult(engine=self.name, failed=True)


class PaddleOcrBackend(OcrBackend):
    name = "paddle"

    def __init__(self, lang: str = "ru"):
        self.lang = lang
        self._ocr = None
        self._dead = False

    def _load(self):
        if self._ocr is None and not self._dead:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(lang=self.lang, use_textline_orientation=True)
            except Exception as e:               # noqa: BLE001
                log.warning("paddleocr недоступен (%s) — этап OCR пропускается", e)
                self._dead = True
        return self._ocr

    def read(self, image: Image.Image) -> OcrResult:
        ocr = self._load()
        if ocr is None:
            return OcrResult(engine=self.name, failed=True)
        try:
            res = ocr.predict(np.asarray(image.convert("RGB")))
            texts: list[str] = []
            for page in res or []:
                # PaddleOCR 3.x отдаёт dict с rec_texts; 2.x — список кортежей.
                if isinstance(page, dict):
                    texts.extend(page.get("rec_texts") or [])
                else:
                    for line in page or []:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            t = line[1]
                            texts.append(t[0] if isinstance(t, (list, tuple)) else str(t))
            return OcrResult(text="\n".join(texts), engine=self.name)
        except Exception as e:                   # noqa: BLE001
            log.warning("paddleocr: ошибка распознавания: %s", e)
            return OcrResult(engine=self.name, failed=True)


# Версия промпта входит в ключ кэша: при смене промпта старые ответы
# невалидны, и молча отдавать их из кэша нельзя.
VLM_PROMPT_VERSION = "v2-brand"

_VLM_PROMPT = (
    "На изображении — фрагмент наружной рекламы из панорамы улицы.\n"
    "Верни СТРОГО JSON без пояснений и без markdown-обёртки, с полями:\n"
    '  "advertiser" — ГЛАВНОЕ ПОЛЕ. Название бренда или рекламодателя. '
    "Крупный логотип и название читаются надёжно — назови их уверенно, "
    "даже если остальной текст мелкий. Если рекламы нет вовсе — null.\n"
    '  "text" — крупный читаемый текст: слоган, название, условия. '
    "Дословно, сохраняя порядок строк. Мелкий нечитаемый шрифт "
    "не восстанавливай — лучше пропусти.\n"
    '  "category" — одна из: недвижимость, авто, финансы, медицина, ретейл, '
    "услуги, развлечения, другое. Если непонятно — null.\n"
    '  "construction" — тип рекламной конструкции, одно из: билборд, '
    "ситиформат, баннер на ограждении, вывеска, стела, экран, "
    "реклама на остановке, другое. Определяй по форме и размещению.\n"
    '  "phone" и "site" — ТОЛЬКО если символы видны крупно и однозначно. '
    "Если шрифт мелкий, размытый или ты не уверен хотя бы в одном символе — "
    "верни null. НЕ достраивай домен по названию бренда и НЕ угадывай цифры: "
    "null здесь правильнее ошибки, неверный телефон хуже отсутствующего.\n"
)


class VlmBackend(OcrBackend):
    """Распознавание через vision-модель. Даёт разом текст, рекламодателя и тему.

    Ключ берётся ТОЛЬКО из окружения (ANTHROPIC_API_KEY) — в конфиг и git он
    не попадает. Отказ API не роняет обход: запись помечается failed и идёт
    дальше, а не обрывает весь прогон.
    """
    name = "vlm-" + VLM_PROMPT_VERSION

    def __init__(self, model: str = "claude-sonnet-4-6", max_side: int = 1024,
                 max_retries: int = 4, timeout: float = 60.0):
        self.model = model
        self.max_side = max_side
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._dead = False

    def _load(self):
        if self._client is None and not self._dead:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                log.error("VLM-движок выбран, но ANTHROPIC_API_KEY не задан — "
                          "распознавание отключено. Ключ берётся только из "
                          "окружения и в конфиг не кладётся")
                self._dead = True
                return None
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout)
            except Exception as e:               # noqa: BLE001
                log.error("SDK anthropic недоступен (%s)", e)
                self._dead = True
        return self._client

    def _encode(self, image: Image.Image) -> str:
        im = image.convert("RGB")
        k = self.max_side / max(im.size)
        if k < 1:                                 # мельче — дешевле по токенам
            im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()

    def read(self, image: Image.Image) -> OcrResult:
        client = self._load()
        if client is None:
            return OcrResult(engine=self.name, failed=True)
        data = self._encode(image)

        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                msg = client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64",
                                                     "media_type": "image/jpeg",
                                                     "data": data}},
                        {"type": "text", "text": _VLM_PROMPT},
                    ]}],
                )
                raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                usage = {"input_tokens": msg.usage.input_tokens,
                         "output_tokens": msg.usage.output_tokens}
                return _parse_vlm(raw, usage, self.name)
            except Exception as e:                # noqa: BLE001
                # Не роняем обход: логируем, ждём с ростом паузы, пробуем снова.
                log.warning("VLM: попытка %d/%d не удалась (%s: %s)",
                            attempt, self.max_retries, type(e).__name__, str(e)[:120])
                if attempt == self.max_retries:
                    break
                time.sleep(delay)
                delay *= 2
        log.error("VLM: все %d попыток исчерпаны — запись помечена как "
                  "нераспознанная, обход продолжается", self.max_retries)
        return OcrResult(engine=self.name, failed=True)


def _parse_vlm(raw: str, usage: dict, engine: str) -> OcrResult:
    """Разбор ответа модели. Модель просили отдать чистый JSON, но обёртку в
    ```json ... ``` терпим — на этом ломаться не за чем."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        s = s[4:] if s.lower().startswith("json") else s
    try:
        d = json.loads(s.strip())
    except Exception:                             # noqa: BLE001
        log.warning("VLM вернул не-JSON (%.80r) — берём как обычный текст", raw[:80])
        return OcrResult(text=raw, engine=engine, usage=usage)
    cat, con = d.get("category"), d.get("construction")
    # Телефон и сайт модель отдаёт отдельными полями — приклеиваем их к тексту,
    # чтобы разбор контактов работал на одном входе. Если модель вернула null,
    # ничего не добавляем: это её отказ угадывать, и он ценнее догадки.
    text = str(d.get("text") or "")
    for extra in (d.get("phone"), d.get("site")):
        if extra and str(extra).lower() not in ("null", "none"):
            text = (text + "\n" + str(extra)).strip()
    return OcrResult(text=text,
                     advertiser=(d.get("advertiser") or None),
                     category=(str(cat) if cat else None),
                     construction=(str(con) if con else None),
                     engine=engine, usage=usage)


def build_backend(cfg) -> OcrBackend:
    name = (cfg.get("ocr.backend", "easyocr") or "easyocr").lower()
    if name == "paddle":
        return PaddleOcrBackend(lang=cfg.get("ocr.paddle_lang", "ru"))
    if name == "vlm":
        return VlmBackend(model=cfg.get("ocr.vlm_model", "claude-sonnet-4-6"),
                          max_side=cfg.get("ocr.vlm_max_side", 1024),
                          max_retries=cfg.get("ocr.vlm_max_retries", 4))
    return EasyOcrBackend(languages=cfg.get("ocr.languages", ["ru", "en"]),
                          gpu=cfg.get("ocr.gpu", False))
