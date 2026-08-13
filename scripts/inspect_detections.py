#!/usr/bin/env python3
"""Диагностика детектора: что OWLv2 видит на панораме и что из этого выживает.

Отвечает на вопрос, который не виден по статистике БД: детектор пропускает щиты
или их отбрасывают фильтры ниже по конвейеру. Для этого показывает ВСЕ детекции,
включая те, что ниже боевого порога, и прогоняет каждую через те же проверки,
что и пайплайн (verify → OCR → категория), печатая, на каком шаге она умерла.

Ничего не пишет в боевую БД и не трогает data/ — только читает панорамы и
складывает размеченные картинки в --out.

    python3 scripts/inspect_detections.py --sample 5 --conf 0.05 --out /tmp/inspect
    python3 scripts/inspect_detections.py --oid <OID> --prompt "a billboard" --prompt "реклама"
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw                                    # noqa: E402

from banner_parser.classify import build_classifier                 # noqa: E402
from banner_parser.config import Config                             # noqa: E402
from banner_parser.detect.detector import _box_to_equirect, _dedup_overlaps  # noqa: E402
from banner_parser.ocr import build_ocr                             # noqa: E402
from banner_parser.verify import build_verifier                     # noqa: E402
from banner_parser.yandex import HttpClient, Panorama               # noqa: E402
from banner_parser.yandex.meta import MetaClient                    # noqa: E402


def pick_oids(db_path: str, n: int, offset: int) -> list[str]:
    """Выбираем уже пройденные oid — на них обход детекцию отработал."""
    c = sqlite3.connect(db_path)
    rows = c.execute(
        "SELECT oid FROM crawl_frontier WHERE done = 1 LIMIT ? OFFSET ?", (n, offset)
    ).fetchall()
    c.close()
    return [r[0] for r in rows]


def detect_all(overview: Image.Image, model, proc, device, prompts: list[str],
               conf: float, n_views: int, fov: float, view_size: int) -> list[dict]:
    """Как OwlDetector.detect, но с порогом из аргументов и с текстом промпта."""
    import torch
    from banner_parser.detect.reproject import horizon_views

    out: list[dict] = []
    for _yaw, view in horizon_views(overview, n_views, fov, out=view_size):
        img = view.image
        inputs = proc(text=[prompts], images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        sizes = torch.tensor([img.size[::-1]]).to(device)
        res = proc.post_process_grounded_object_detection(
            outputs, threshold=conf, target_sizes=sizes)[0]
        for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(),
                                     res["labels"].tolist()):
            x0, y0, x1, y1 = (int(v) for v in box)
            det = _box_to_equirect(view.uv_map, x0, y0, x1, y1, float(score))
            if det is not None:
                out.append({"det": det, "score": float(score),
                            "prompt": prompts[int(label)]})
    # Дедуп по IoU — тот же, что в бою; сохраняем привязку промпта к боксу.
    kept = _dedup_overlaps([d["det"] for d in out])
    by_key = {(round(d["det"].fx0, 6), round(d["det"].fy0, 6)): d for d in out}
    return [by_key[(round(k.fx0, 6), round(k.fy0, 6))] for k in kept
            if (round(k.fx0, 6), round(k.fy0, 6)) in by_key]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--oid", action="append", default=[], help="конкретные oid панорам")
    ap.add_argument("--sample", type=int, default=0, help="взять N пройденных панорам из БД")
    ap.add_argument("--offset", type=int, default=0, help="смещение выборки")
    ap.add_argument("--conf", type=float, default=None, help="порог показа (ниже боевого — видно пропуски)")
    ap.add_argument("--prompt", action="append", default=[], help="свой набор промптов")
    ap.add_argument("--out", default="/tmp/inspect", help="куда класть размеченные панорамы")
    ap.add_argument("--ocr", action="store_true", help="гонять OCR (медленно, но показывает текст)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    live_conf = cfg.get("detector.conf", 0.22)
    show_conf = args.conf if args.conf is not None else live_conf
    only_cat = cfg.get("filter.only_category", None)

    http = HttpClient(cfg)
    meta = MetaClient(http)
    verifier = build_verifier(cfg)
    classifier = build_classifier(cfg)
    ocr = build_ocr(cfg) if args.ocr else None

    oids = list(args.oid)
    if args.sample:
        oids += pick_oids(cfg.get("storage.db_path", "data/banners.sqlite"),
                          args.sample, args.offset)
    if not oids:
        ap.error("укажите --oid или --sample")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    import torch

    model_name = cfg.get("detector.owl_model", "google/owlv2-base-patch16-ensemble")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()

    from banner_parser.detect.detector import OwlDetector
    prompts = args.prompt or OwlDetector.DEFAULT_PROMPTS
    print(f"промпты ({len(prompts)}): {prompts}")
    print(f"боевой порог={live_conf}, показываю от {show_conf}, фильтр темы={only_cat}\n")

    totals = {"детекций": 0, "выше_порога": 0, "прошло_verify": 0, "прошло_фильтр": 0}

    for oid in oids:
        raw = meta.by_oid(oid)
        if raw is None:
            print(f"[{oid[:20]}] meta не отдала данные — пропуск")
            continue
        ref = meta.parse(raw)
        pano = Panorama(ref, http, cfg.get("panorama.tile_workers", 16))
        overview = pano.stitch(cfg.get("panorama.overview_zoom", 2))

        found = detect_all(overview, model, proc, device, prompts, show_conf,
                           n_views=6, fov=90.0, view_size=960)
        found.sort(key=lambda d: d["score"], reverse=True)

        print(f"=== {ref.panoid}  ({ref.lon:.5f},{ref.lat:.5f})  "
              f"детекций≥{show_conf}: {len(found)} ===")
        canvas = overview.copy()
        draw = ImageDraw.Draw(canvas)
        W, H = canvas.size

        for i, d in enumerate(found[:12]):
            det, score = d["det"], d["score"]
            box = (det.fx0 * W, det.fy0 * H, det.fx1 * W, det.fy1 * H)
            above = score >= live_conf
            draw.rectangle(box, outline=(0, 255, 0) if above else (255, 210, 0), width=4)
            draw.text((box[0] + 5, box[1] + 5), f"{i}:{score:.2f}",
                      fill=(0, 255, 0) if above else (255, 210, 0))

            totals["детекций"] += 1
            verdict = []
            if not above:
                verdict.append(f"НИЖЕ ПОРОГА ({score:.3f}<{live_conf})")
            else:
                totals["выше_порога"] += 1
                crop = overview.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                vr = verifier.verify(crop) if verifier else None
                if vr is not None and not vr.is_ad:
                    verdict.append(f"ОТСЕЯН verify [{vr.reason}]")
                else:
                    totals["прошло_verify"] += 1
                    text = ocr.read(crop) if ocr else ""
                    cat = classifier.classify(text, crop)
                    if only_cat and cat != only_cat:
                        verdict.append(f"ОТСЕЯН фильтром темы (='{cat}')")
                    else:
                        totals["прошло_фильтр"] += 1
                        verdict.append(f"СОХРАНИЛСЯ БЫ (тема '{cat}')")
                    if ocr:
                        verdict.append(f"OCR={text[:60]!r}")
            print(f"  #{i} score={score:.3f} промпт={d['prompt']!r:45} "
                  f"fx=[{det.fx0:.3f}..{det.fx1:.3f}] fy=[{det.fy0:.3f}..{det.fy1:.3f}]")
            print(f"      → {' | '.join(verdict)}")

        path = out_dir / f"{ref.panoid}.jpg"
        canvas.save(path, quality=85)
        print(f"  размеченная панорама: {path}\n")

    print("ИТОГО:", ", ".join(f"{k}={v}" for k, v in totals.items()))


if __name__ == "__main__":
    main()
