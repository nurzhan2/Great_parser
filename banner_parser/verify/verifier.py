"""Проверка, что кроп — действительно рекламный баннер, а не стена/небо/дорога.

Два бэкенда:
  heuristic — без зависимостей: рекламный баннер яркий (насыщенный/цветной) и
              содержит много текста/графики (высокая плотность контуров).
              Пустые стены, небо, асфальт — низкие по этим метрикам и отсеиваются.
  clip      — zero-shot: сравнивает кроп с промптами «реклама» vs фон (open-clip).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    is_ad: bool
    score: float
    reason: str


def ad_metrics(image: Image.Image) -> dict:
    """Быстрые метрики «рекламности» кропа."""
    im = image.convert("RGB")
    im.thumbnail((512, 512))
    a = np.asarray(im).astype(np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]

    # Цветность (Hasler–Süsstrunk)
    rg, yb = R - G, 0.5 * (R + G) - B
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    # Средняя насыщенность (HSV S)
    mx, mn = a.max(-1), a.min(-1)
    saturation = float(((mx - mn) / (mx + 1e-6)).mean())

    # Плотность контуров = доля пикселей с сильным градиентом (текст/графика)
    gray = a.mean(-1)
    gray_std = float(gray.std())
    bright_ratio = float((gray >= 185.0).mean())
    dark_ratio = float((gray <= 100.0).mean())
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_density = float(((gx > 24).mean() + (gy > 24).mean()) / 2)
  
     return {
    "colorfulness": colorfulness,
    "saturation": saturation,
    "edge_density": edge_density,
    "gray_std": gray_std,
    "bright_ratio": bright_ratio,
    "dark_ratio": dark_ratio,}


class AdVerifier:
    def verify(self, image: Image.Image, text: str = "") -> VerifyResult:
        raise NotImplementedError


class HeuristicVerifier(AdVerifier):
    def __init__(self, min_colorfulness: float = 18.0, min_edge_density: float = 0.040,
                 min_saturation: float = 0.12, paper_min_bright_ratio: float = 0.45, paper_min_dark_ratio: float = 0.01, paper_min_contrast: float = 18.0,):
        self.min_colorfulness = min_colorfulness
        self.min_edge_density = min_edge_density
        self.min_saturation = min_saturation
        self.paper_min_bright_ratio = paper_min_bright_ratio
        self.paper_min_dark_ratio = paper_min_dark_ratio
        self.paper_min_contrast = paper_min_contrast

    def verify(self, image: Image.Image, text: str = "") -> VerifyResult:
        m = ad_metrics(image)
        # Наличие распознанного текста — сильный признак рекламы.
        has_text = len((text or "").strip()) >= 4
        colorful = m["colorfulness"] >= self.min_colorfulness or m["saturation"] >= self.min_saturation
        textured = m["edge_density"] >= self.min_edge_density
        paper_like = (
            textured
            and m["bright_ratio"] >= self.paper_min_bright_ratio
            and m["dark_ratio"] >= self.paper_min_dark_ratio
            and m["gray_std"] >= self.paper_min_contrast
        )
        is_ad = (
            (textured and colorful)
            or paper_like
            or (has_text and textured)
        )

        # score: насколько уверенно превышены пороги
        score = min(1.0, 0.5 * m["edge_density"] / self.min_edge_density
                    + 0.5 * m["colorfulness"] / self.min_colorfulness)
        reason = (f"edges={m['edge_density']:.3f} colorful={m['colorfulness']:.1f} "
                  f"sat={m['saturation']:.2f} text={'да' if has_text else 'нет'}")
        return VerifyResult(is_ad, round(score, 3), reason)


class ClipVerifier(AdVerifier):
    """Zero-shot реклама vs фон через open-clip. Fallback к эвристике."""

    AD_PROMPTS = ["рекламный баннер", "рекламный щит с текстом", "вывеска с рекламой"]
    BG_PROMPTS = ["глухая стена здания", "небо с облаками", "асфальтовая дорога",
                  "деревья и кусты", "пустой забор"]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = self._pre = self._tok = None
        self._fallback = HeuristicVerifier()

    def _load(self):
        if self._model is None:
            import open_clip  # lazy
            self._model, _, self._pre = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            self._tok = open_clip.get_tokenizer("ViT-B-32")
        return self._model

    def verify(self, image: Image.Image, text: str = "") -> VerifyResult:
        try:
            import torch
            model = self._load()
            prompts = self.AD_PROMPTS + self.BG_PROMPTS
            img = self._pre(image.convert("RGB")).unsqueeze(0)
            toks = self._tok(prompts)
            with torch.no_grad():
                imf = model.encode_image(img)
                txf = model.encode_text(toks)
                imf /= imf.norm(dim=-1, keepdim=True)
                txf /= txf.norm(dim=-1, keepdim=True)
                probs = (100 * imf @ txf.T).softmax(dim=-1)[0]
            ad_p = float(probs[:len(self.AD_PROMPTS)].sum())
            return VerifyResult(ad_p >= self.threshold, round(ad_p, 3),
                                f"clip P(реклама)={ad_p:.2f}")
        except Exception as e:  # noqa: BLE001
            log.warning("CLIP недоступен (%s) — эвристика", e)
            return self._fallback.verify(image, text)


def build_verifier(cfg) -> Optional[AdVerifier]:
    if not cfg.get("verify.enabled", True):
        return None
    if cfg.get("verify.backend", "heuristic") == "clip":
        return ClipVerifier(threshold=cfg.get("verify.clip_threshold", 0.5))
    return HeuristicVerifier(
        min_colorfulness=cfg.get("verify.min_colorfulness", 18.0),
        min_edge_density=cfg.get("verify.min_edge_density", 0.040),
        min_saturation=cfg.get("verify.min_saturation", 0.12),
        paper_min_bright_ratio=cfg.get(
            "verify.paper_min_bright_ratio", 0.45
        ),
        paper_min_dark_ratio=cfg.get(
            "verify.paper_min_dark_ratio", 0.01
        ),
        paper_min_contrast=cfg.get(
            "verify.paper_min_contrast", 18.0
        ),
    )
