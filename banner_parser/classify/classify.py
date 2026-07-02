"""Классификация темы баннера: keyword-baseline по тексту OCR или CLIP по картинке."""
from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

log = logging.getLogger(__name__)

# Ключевые слова по категориям (нижний регистр, по подстроке).
KEYWORDS: dict[str, list[str]] = {
    "недвижимость": ["жк ", "жилой комплекс", "квартир", "новостро", "застройщик",
                     "ипотек", "апартамент", "недвижим", "м²", "кв.м", "от застройщика",
                     "продажа квартир", "таунхаус", "коттедж", "риэлт", "квартал",
                     "аренда", "аренду", "снять", "сдам", "сдаёт", "сдает", "сдаётся",
                     "сдается", "продажа", "продаётся", "продается", "продам", "куплю",
                     "студи", "комнат", "участок", "заселение", "сдача", "монолит",
                     "офис продаж", "дом от", "квартиры от", "заезжай"],
    "авто": ["автосалон", "автосервис", "шиномонтаж", "запчаст", "кредит на авто",
             "лада", "toyota", "kia", "автомобил", "шины", "автомойка"],
    "финансы": ["банк", "кредит", "займ", "вклад", "страхов", "микрозайм", "рассрочк",
                "инвест", "ипотека под"],
    "медицина": ["клиник", "стоматолог", "медицин", "аптек", "здоровь", "лечени",
                 "диагностик", "анализы"],
    "ретейл": ["магазин", "скидк", "распродаж", "супермаркет", "sale", "акция",
               "торговый центр", "тц "],
    "развлечения": ["концерт", "театр", "кино", "фестивал", "парк", "шоу", "билеты"],
    "услуги": ["ремонт", "услуги", "юрист", "образован", "курсы", "салон красоты",
               "клининг", "доставка"],
}


class Classifier:
    def classify(self, text: str, image: Optional[Image.Image] = None) -> str:
        raise NotImplementedError


class KeywordClassifier(Classifier):
    def __init__(self, categories: Optional[list[str]] = None):
        self.categories = categories

    def classify(self, text: str, image: Optional[Image.Image] = None) -> str:
        low = (text or "").lower()
        best, best_hits = "другое", 0
        for cat, words in KEYWORDS.items():
            hits = sum(1 for w in words if w in low)
            if hits > best_hits:
                best, best_hits = cat, hits
        return best


class ClipClassifier(Classifier):
    """Zero-shot по картинке через open-clip. Fallback к keyword по тексту."""

    def __init__(self, categories: list[str]):
        self.categories = categories
        self._model = None
        self._pre = None
        self._tok = None
        self._kw = KeywordClassifier(categories)

    def _load(self):
        if self._model is None:
            import open_clip  # lazy
            import torch  # noqa: F401
            self._model, _, self._pre = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            self._tok = open_clip.get_tokenizer("ViT-B-32")
        return self._model

    def classify(self, text: str, image: Optional[Image.Image] = None) -> str:
        if image is None:
            return self._kw.classify(text)
        try:
            import torch
            model = self._load()
            prompts = [f"рекламный баннер на тему: {c}" for c in self.categories]
            img = self._pre(image).unsqueeze(0)
            toks = self._tok(prompts)
            with torch.no_grad():
                imf = model.encode_image(img)
                txf = model.encode_text(toks)
                imf /= imf.norm(dim=-1, keepdim=True)
                txf /= txf.norm(dim=-1, keepdim=True)
                sims = (imf @ txf.T).softmax(dim=-1)[0]
            return self.categories[int(sims.argmax())]
        except Exception as e:  # noqa: BLE001
            log.warning("CLIP недоступен (%s) — откат к keyword", e)
            return self._kw.classify(text)


def build_classifier(cfg) -> Classifier:
    cats = cfg.get("classify.categories", [])
    if cfg.get("classify.backend", "keyword") == "clip":
        return ClipClassifier(cats)
    return KeywordClassifier(cats)
