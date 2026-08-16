#!/usr/bin/env python3
"""Сравнение OCR-движков на одном наборе кропов с ручным эталоном.

Метрика для текста — доля слов эталона (≥3 символов), найденных в
распознанном. Для VLM дополнительно считается точность по рекламодателю и
теме и реальная стоимость по токенам из ответов API.

    python3 scripts/bench_ocr.py --crops DIR --gt ocr_gt_full.json \
        --engine easyocr --engine paddle --engine vlm
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image                                              # noqa: E402

from banner_parser.config import Config                            # noqa: E402
from banner_parser.ocr.engines import build_backend                # noqa: E402

# Цена за 1M токенов, $. Источник — прайс Anthropic для claude-sonnet-4-6.
PRICE_IN, PRICE_OUT = 3.00, 15.00


def norm(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я%]+", " ", s).strip()


def text_score(gt: str, got: str):
    words = [w for w in norm(gt).split() if len(w) >= 3]
    hay = norm(got)
    return (sum(1 for w in words if w in hay) / len(words)) if words else None


def match(expected, got) -> bool:
    if not expected:
        return got in (None, "")          # ожидали пусто — пусто и есть верно
    return bool(got) and norm(str(expected)) in norm(str(got))


class _Cfg(Config):
    def __init__(self, backend: str):
        super().__init__({"ocr": {"backend": backend, "languages": ["ru", "en"],
                                  "gpu": False}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--engine", action="append", required=True)
    args = ap.parse_args()

    gt = json.load(io.open(args.gt, encoding="utf-8"))
    files = sorted(gt.keys())
    print(f"кропов в эталоне: {len(files)}\n")
    print(f"{'движок':<12}{'текст':>8}{'рекламод.':>11}{'тема':>8}"
          f"{'с/кроп':>9}{'$/100k':>10}")

    for name in args.engine:
        backend = build_backend(_Cfg(name))
        t0 = time.time()
        scores, adv_ok, cat_ok, tok_in, tok_out, done = [], 0, 0, 0, 0, 0
        for fn in files:
            im = Image.open(os.path.join(args.crops, fn)).convert("RGB")
            r = backend.read(im)
            if r.failed:
                continue
            done += 1
            s = text_score(gt[fn]["text"], r.text)
            if s is not None:
                scores.append(s)
            if match(gt[fn].get("advertiser"), r.advertiser):
                adv_ok += 1
            if match(gt[fn].get("category"), r.category):
                cat_ok += 1
            tok_in += r.usage.get("input_tokens", 0)
            tok_out += r.usage.get("output_tokens", 0)
        dt = time.time() - t0
        if not done:
            print(f"{name:<12}  движок не отработал ни одного кропа")
            continue
        cost = ((tok_in * PRICE_IN + tok_out * PRICE_OUT) / 1e6 / done * 100_000
                if tok_in else 0.0)
        print(f"{name:<12}{100*sum(scores)/len(scores):>7.1f}%"
              f"{100*adv_ok/done:>10.0f}%{100*cat_ok/done:>7.0f}%"
              f"{dt/done:>9.1f}{cost:>10.0f}")
        if tok_in:
            print(f"{'':<12}токенов: вход {tok_in}, выход {tok_out} на {done} кропов")


if __name__ == "__main__":
    main()
