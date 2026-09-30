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
    is_realty: Optional[str] = None       # да | нет | не уверен
    developer: Optional[str] = None       # застройщик
    complex_name: Optional[str] = None    # название ЖК
    offer_type: Optional[str] = None      # новостройка | аренда | ипотека | ...
    advertiser_type: Optional[str] = None # застройщик | агентство | банк | частное лицо
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
            is_realty TEXT, developer TEXT, complex_name TEXT,
            offer_type TEXT, advertiser_type TEXT,
            usage TEXT, created REAL)""")
        # CREATE TABLE IF NOT EXISTS не трогает существующую таблицу, поэтому
        # старый кэш остаётся без новых колонок и падает на SELECT.
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(ocr_cache)")}
        for col in ("construction", "is_realty", "developer", "complex_name",
                    "offer_type", "advertiser_type"):
            if col not in have:
                self.conn.execute(f"ALTER TABLE ocr_cache ADD COLUMN {col} TEXT")
        self.conn.commit()
        self.hits = self.misses = 0

    @staticmethod
    def key(image: Image.Image, engine: str) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG", compress_level=1)
        return engine + ":" + hashlib.sha256(buf.getvalue()).hexdigest()

    def get(self, k: str) -> Optional[OcrResult]:
        row = self.conn.execute(
            "SELECT engine, text, advertiser, category, construction, is_realty, "
            "developer, complex_name, offer_type, advertiser_type, usage "
            "FROM ocr_cache WHERE key=?",
            (k,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return OcrResult(text=row[1] or "", advertiser=row[2], category=row[3],
                         construction=row[4], is_realty=row[5], developer=row[6],
                         complex_name=row[7], offer_type=row[8],
                         advertiser_type=row[9], engine=row[0],
                         usage=json.loads(row[10] or "{}"))

    def put(self, k: str, r: OcrResult) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO ocr_cache "
            "(key, engine, text, advertiser, category, construction, is_realty, "
            "developer, complex_name, offer_type, advertiser_type, usage, created) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (k, r.engine, r.text, r.advertiser, r.category, r.construction,
             r.is_realty, r.developer, r.complex_name, r.offer_type,
             r.advertiser_type, json.dumps(r.usage), time.time()))
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
VLM_PROMPT_VERSION = "v3-realty"

_VLM_PROMPT = (
    "На изображении — фрагмент наружной рекламы из панорамы улицы.\n"
    "Нас интересует ТОЛЬКО реклама недвижимости: застройщики, жилые "
    "комплексы, агентства недвижимости, продажа и аренда жилой и "
    "коммерческой недвижимости, ипотечные предложения банков.\n"
    "Верни СТРОГО JSON без пояснений и без markdown-обёртки, с полями:\n"
    '  "is_realty" — "да", "нет" или "не уверен". Реклама магазина, кафе, '
    "автосервиса, кино, банка без ипотеки — это \"нет\".\n"
    '  "advertiser" — название бренда или рекламодателя как написано.\n'
    '  "construction" — тип конструкции: билборд, ситиформат, баннер на '
    "ограждении, вывеска, стела, экран, реклама на остановке, другое.\n"
    '  "text" — крупный читаемый текст дословно. Мелкий нечитаемый шрифт '
    "не восстанавливай.\n"
    "Если is_realty = \"да\", дополнительно:\n"
    '  "developer" — застройщик или компания-рекламодатель;\n'
    '  "complex_name" — название жилого комплекса (ЖК), если указано. '
    "Это ОТДЕЛЬНАЯ сущность от застройщика: «Квартал Домашний» — это ЖК, "
    "а застройщик у него «Самолёт».\n"
    '  "offer_type" — одно из: новостройка, вторичка, аренда жилая, '
    "аренда коммерческая, ипотека, загородная;\n"
    '  "advertiser_type" — одно из: застройщик, агентство, банк, частное лицо. '
    "Частное лицо — это «сдам»/«продам» с личным номером и без юрлица.\n"
    '  "phone" и "site" — ТОЛЬКО если символы видны крупно и однозначно. '
    "Если шрифт мелкий, размытый или ты не уверен хотя бы в одном символе — "
    "верни null. НЕ достраивай домен по названию бренда и НЕ угадывай цифры: "
    "null здесь правильнее ошибки, неверный телефон хуже отсутствующего.\n"
)


class CloudVlmBackend(OcrBackend):
    """Общая механика облачных vision-движков: кодирование картинки, ретраи с
    ростом паузы, разбор JSON и правило «неустранимое не повторяем».

    Наследники реализуют только _call() — сам запрос к своему API. Промпт,
    парсер и политика ошибок общие: промпт v3 единственный, что убрал
    выдуманные телефоны, и расходиться между провайдерами он не должен.
    """
    name = "cloud"
    env_key = ""

    def __init__(self, model: str, max_side: int = 1024,
                 max_retries: int = 4, timeout: float = 60.0):
        self.model = model
        self.max_side = max_side
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._dead = False

    def _make_client(self, key: str):
        raise NotImplementedError

    def _call(self, client, data_b64: str) -> tuple[str, dict]:
        """Вернуть (сырой текст ответа, usage)."""
        raise NotImplementedError

    def _load(self):
        if self._client is None and not self._dead:
            key = os.environ.get(self.env_key)
            if not key:
                log.error("движок %s выбран, но %s не задан — распознавание "
                          "отключено. Ключ берётся только из окружения",
                          self.name, self.env_key)
                self._dead = True
                return None
            try:
                self._client = self._make_client(key)
            except Exception as e:               # noqa: BLE001
                log.error("SDK для %s недоступен (%s)", self.name, e)
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
                raw, usage = self._call(client, data)
                return _parse_vlm(raw, usage, self.name)
            except Exception as e:                # noqa: BLE001
                msg = str(e)[:160]
                # Нехватка средств, неверный ключ, отказ по правам повтором не
                # лечатся, а каждый повтор — лишний вызов. Ретраим только
                # временное: лимиты, таймауты, 5xx, обрывы сети.
                code = getattr(getattr(e, "response", None), "status_code", None)
                fatal = code in (400, 401, 403, 404) and "rate" not in msg.lower()
                log.warning("%s: попытка %d/%d не удалась (%s: %s)",
                            self.name, attempt, self.max_retries, type(e).__name__, msg)
                if fatal:
                    log.error("%s: ошибка неустранима повтором (HTTP %s) — "
                              "прекращаем попытки и больше не ходим в API. "
                              "Обход продолжится с запасным движком", self.name, code)
                    self._dead = True
                    break
                if attempt == self.max_retries:
                    break
                time.sleep(delay)
                delay *= 2
        log.error("%s: попытки исчерпаны — запись помечена нераспознанной", self.name)
        return OcrResult(engine=self.name, failed=True)


class VlmBackend(CloudVlmBackend):
    """Anthropic. Ключ — только из ANTHROPIC_API_KEY."""
    name = "vlm-" + VLM_PROMPT_VERSION
    env_key = "ANTHROPIC_API_KEY"

    def __init__(self, model: str = "claude-sonnet-4-6", max_side: int = 1024,
                 max_retries: int = 4, timeout: float = 60.0):
        super().__init__(model, max_side, max_retries, timeout)

    def _make_client(self, key: str):
        import anthropic
        return anthropic.Anthropic(api_key=key, timeout=self.timeout)

    def _call(self, client, data_b64: str):
        msg = client.messages.create(
            model=self.model, max_tokens=1500,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg",
                                             "data": data_b64}},
                {"type": "text", "text": _VLM_PROMPT},
            ]}])
        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return raw, {"input_tokens": msg.usage.input_tokens,
                     "output_tokens": msg.usage.output_tokens}


class OpenAiBackend(CloudVlmBackend):
    """OpenAI. Промпт, парсер и политика ошибок — те же, что у Anthropic:
    расхождение промптов между провайдерами вернуло бы выдуманные контакты.
    Ключ — только из OPENAI_API_KEY."""
    name = "openai-" + VLM_PROMPT_VERSION
    env_key = "OPENAI_API_KEY"

    def __init__(self, model: str = "gpt-4o-mini", max_side: int = 1024,
                 max_retries: int = 4, timeout: float = 60.0):
        super().__init__(model, max_side, max_retries, timeout)

    def _make_client(self, key: str):
        import openai
        return openai.OpenAI(api_key=key, timeout=self.timeout)

    def _call(self, client, data_b64: str):
        r = client.chat.completions.create(
            model=self.model, max_tokens=1500,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _VLM_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + data_b64}},
            ]}])
        raw = r.choices[0].message.content or ""
        u = getattr(r, "usage", None)
        return raw, {"input_tokens": getattr(u, "prompt_tokens", 0),
                     "output_tokens": getattr(u, "completion_tokens", 0)}


class GeminiBackend(CloudVlmBackend):
    """Google Gemini. Промпт, парсер и политика ошибок — те же, что у Anthropic
    и OpenAI: расхождение промптов между провайдерами вернуло бы выдуманные
    контакты. Ключ — только из GEMINI_API_KEY.

    Смысл движка — бесплатный тариф Gemini Developer API: Flash-модели там
    доступны без оплаты, зрение полноценное (в отличие от DeepSeek, где
    картинка режется до 384 токенов и мелкий текст не выживает).
    Лимиты бесплатного тарифа — порядка 10 запросов в минуту и ~1500 в сутки,
    поэтому при 12 вызовах на панораму выходит около сотни панорам в день.
    Ошибку 429 базовый класс уже трактует как повторяемую и ждёт с ростом
    паузы, так что упор в лимит замедляет обход, но не роняет его.
    """
    name = "gemini-" + VLM_PROMPT_VERSION
    env_key = "GEMINI_API_KEY"

    def __init__(self, model: str = "gemini-2.5-flash", max_side: int = 1024,
                 max_retries: int = 4, timeout: float = 60.0):
        super().__init__(model, max_side, max_retries, timeout)

    def _make_client(self, key: str):
        # Новый SDK google-genai; старый google-generativeai не поддерживается.
        from google import genai
        return genai.Client(api_key=key)

    def _call(self, client, data_b64: str):
        import base64
        from google.genai import types
        r = client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=base64.b64decode(data_b64),
                                      mime_type="image/jpeg"),
                _VLM_PROMPT,
            ],
            config=types.GenerateContentConfig(max_output_tokens=1500))
        raw = r.text or ""
        u = getattr(r, "usage_metadata", None)
        return raw, {"input_tokens": getattr(u, "prompt_token_count", 0) or 0,
                     "output_tokens": getattr(u, "candidates_token_count", 0) or 0}


class DeepSeekBackend(CloudVlmBackend):
    """DeepSeek V4-Flash-Vision. Самый дешёвый из доступных с российского
    хостинга: ~$0.22 за 1M входных токенов вне пика против $1 у Haiku 4.5.

    Ограничение и обход. DeepSeek ужимает каждую картинку до бюджета примерно
    800x800 пикселей и берёт максимум 384 токена за штуку. Для щита с
    телефоном это мало: 384 токена на весь кадр — вчетверо грубее, чем даёт
    Claude на том же кропе, и мелкие цифры в такой сетке смазываются.

    Но потолок применяется К КАЖДОЙ картинке отдельно, а в один запрос их
    влезает до 600. Поэтому крупный кроп режем на плитки: 2x2 с перехлёстом
    дают 4x384 токена вместо 384, и каждая плитка укладывается в бюджет
    800x800 без ужатия, то есть идёт в родном разрешении.

    Перехлёст обязателен: телефон, попавший на стык плиток, иначе разорвётся
    пополам и не соберётся ни в одной. При 15% один и тот же номер целиком
    виден хотя бы в одной плитке.

    Мелкие кропы не режем — они и так проходят под бюджет, а нарезка только
    добавила бы вызовов. Обзорную картинку целиком шлём всегда: по плиткам
    модель не поймёт, что это один щит, и придумает несколько объявлений.
    """
    name = "deepseek-" + VLM_PROMPT_VERSION
    env_key = "DEEPSEEK_API_KEY"

    BUDGET_PX = 640_000        # ~800x800: выше этого DeepSeek ужимает картинку
    OVERLAP = 0.15             # перехлёст плиток
    # Ниже этой стороны резать нечего: плитки выйдут меньше 80 пикселей.
    # Порог по площади был ошибкой: кропы 391x303 и 359x468 уходили одной
    # картинкой и возвращали пустоту там, где Sonnet читал текст. Мелкому
    # кропу нарезка нужна не меньше — DeepSeek дотягивает каждую плитку до
    # 384x384, то есть мелкий текст идёт с увеличением, а не с ужатием.
    TILE_MIN_SIDE = 160

    def __init__(self, model: str = "deepseek-v4-flash-vision-exp",
                 max_side: int = 1024, max_retries: int = 4,
                 timeout: float = 90.0, tile: bool = True):
        super().__init__(model, max_side, max_retries, timeout)
        self.tile = tile

    def _make_client(self, key: str):
        # API совместим с OpenAI, отдельный SDK не нужен.
        from openai import OpenAI
        return OpenAI(api_key=key, base_url="https://api.deepseek.com",
                      timeout=self.timeout)

    @staticmethod
    def _b64(im) -> str:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    def _encode(self, image):
        """Список base64: обзорная картинка + плитки. Базовый read() передаёт
        результат в _call() не разбирая, поэтому список тут допустим."""
        im = image.convert("RGB")

        # Обзорная — ужимаем как обычно, её задача дать модели общий вид.
        overview = im
        k = self.max_side / max(im.size)
        if k < 1:
            overview = im.resize((max(1, int(im.width * k)),
                                  max(1, int(im.height * k))), Image.LANCZOS)
        out = [self._b64(overview)]

        w, h = im.size
        if not self.tile or min(w, h) < self.TILE_MIN_SIDE:
            return out                     # совсем мелкий кроп резать бессмысленно

        ox, oy = int(w * self.OVERLAP / 2), int(h * self.OVERLAP / 2)
        for row in (0, 1):
            for col in (0, 1):
                left = max(0, col * w // 2 - ox)
                top = max(0, row * h // 2 - oy)
                right = min(w, (col + 1) * w // 2 + ox)
                bottom = min(h, (row + 1) * h // 2 + oy)
                out.append(self._b64(im.crop((left, top, right, bottom))))
        log.debug("%s: кроп %dx%d -> %d изображений", self.name, w, h, len(out))
        return out

    def _call(self, client, data):
        # Пояснение про плитки идёт ОТДЕЛЬНЫМ текстом перед общим промптом:
        # сам промпт v3 един для всех провайдеров и не правится.
        content = []
        if isinstance(data, str):
            data = [data]
        if len(data) > 1:
            content.append({"type": "text", "text":
                "Ниже одно и то же рекламное изображение: сначала целиком, "
                "затем его фрагменты крупным планом с перехлёстом. Это ОДИН "
                "объект, а не несколько. Используй фрагменты, чтобы разобрать "
                "мелкий текст и цифры телефона, и ответь одним JSON."})
        for b in data:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
        content.append({"type": "text", "text": _VLM_PROMPT})

        r = client.chat.completions.create(
            model=self.model, max_tokens=1500,
            messages=[{"role": "user", "content": content}],
            # Размышление включено по умолчанию на высоком уровне и жжёт токены,
            # а для чтения текста с картинки оно не нужно.
            extra_body={"thinking": {"type": "disabled"}})
        raw = r.choices[0].message.content or ""
        u = getattr(r, "usage", None)
        return raw, {"input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                     "output_tokens": getattr(u, "completion_tokens", 0) or 0}


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
    g = lambda k: (str(d.get(k)).strip() if d.get(k) else None)
    return OcrResult(text=text,
                     advertiser=(d.get("advertiser") or None),
                     category=(str(cat) if cat else None),
                     construction=(str(con) if con else None),
                     is_realty=g("is_realty"), developer=g("developer"),
                     complex_name=g("complex_name"), offer_type=g("offer_type"),
                     advertiser_type=g("advertiser_type"),
                     engine=engine, usage=usage)


def build_backend(cfg) -> OcrBackend:
    name = (cfg.get("ocr.backend", "easyocr") or "easyocr").lower()
    if name == "paddle":
        return PaddleOcrBackend(lang=cfg.get("ocr.paddle_lang", "ru"))
    if name == "openai":
        return OpenAiBackend(model=cfg.get("ocr.openai_model", "gpt-4o-mini"),
                             max_side=cfg.get("ocr.vlm_max_side", 1024),
                             max_retries=cfg.get("ocr.vlm_max_retries", 4))
    if name == "deepseek":
        return DeepSeekBackend(
            model=cfg.get("ocr.deepseek_model", "deepseek-v4-flash-vision-exp"),
            max_side=cfg.get("ocr.vlm_max_side", 1024),
            max_retries=cfg.get("ocr.vlm_max_retries", 4),
            tile=cfg.get("ocr.deepseek_tile", True))
    if name == "gemini":
        return GeminiBackend(model=cfg.get("ocr.gemini_model", "gemini-2.5-flash"),
                             max_side=cfg.get("ocr.vlm_max_side", 1024),
                             max_retries=cfg.get("ocr.vlm_max_retries", 4))
    if name == "vlm":
        return VlmBackend(model=cfg.get("ocr.vlm_model", "claude-sonnet-4-6"),
                          max_side=cfg.get("ocr.vlm_max_side", 1024),
                          max_retries=cfg.get("ocr.vlm_max_retries", 4))
    return EasyOcrBackend(languages=cfg.get("ocr.languages", ["ru", "en"]),
                          gpu=cfg.get("ocr.gpu", False))
