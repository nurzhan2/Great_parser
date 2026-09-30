"""Сравнение бесплатных VLM-OCR на кропах с ИЗВЕСТНЫМИ телефонами.

Эталон — то, что DeepSeek уже вытащил верно. Каждая модель грузится
отдельно и в try: падение одной не должно уносить весь тест.
Меряем главное: угадан ли телефон целиком и за сколько секунд.
"""
import os, sqlite3, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)
for n in ("paddleocr", "paddlex", "transformers", "PIL"):
    logging.getLogger(n).setLevel(logging.ERROR)
from PIL import Image
from banner_parser.config import Config
from banner_parser.ocr import extract_contacts

cfg = Config.load(None)
c = sqlite3.connect(cfg.get("storage.db_path", "data/banners.sqlite"))
rows = c.execute("""select crop_image_path, phones from banners
                    where phones is not null and phones<>'' and phones<>'[]'
                    and crop_image_path is not null order by rowid desc limit 25""").fetchall()
SAMPLES = [(p, ph) for p, ph in rows if p and os.path.exists(p)][:6]
print(f"кропов: {len(SAMPLES)}")
for p, ph in SAMPLES:
    print(f"   {os.path.basename(p)} -> {ph}")
print()

PROMPT = ("Извлеки весь текст с изображения. Особенно точно перепиши все цифры "
          "телефонных номеров. Ответь только текстом, без пояснений.")


def score(name, fn):
    """Прогон одной модели по всем кропам."""
    print(f"\n{'='*60}\n### {name}\n{'='*60}")
    ok = 0
    for path, ref in SAMPLES:
        img = Image.open(path).convert("RGB")
        t0 = time.monotonic()
        try:
            txt = fn(img, path) or ""
        except Exception as e:
            print(f"  {os.path.basename(path)[:28]}: ОШИБКА {type(e).__name__}: {str(e)[:90]}")
            continue
        dt = time.monotonic() - t0
        ph = extract_contacts(txt).phones
        hit = bool(ph) and all(p in ref for p in ph)
        ok += hit
        print(f"  {os.path.basename(path)[:28]:30} {dt:6.1f}с  {str(ph)[:34]:36}"
              f"{'ВЕРНО' if hit else ('НЕВЕРНО' if ph else '')}")
        if txt.strip():
            print(f"      {txt[:80]!r}")
    print(f"\n>>> ИТОГ {name}: {ok}/{len(SAMPLES)}")
    return ok


results = {}

# ---------- 1. PaddleOCR-VL 0.9B ----------
try:
    from paddleocr import PaddleOCRVL
    pvl = PaddleOCRVL()

    def run_pvl(img, path):
        out = pvl.predict(path)
        parts = []
        for res in out:
            d = res if isinstance(res, dict) else getattr(res, "json", {}) or {}
            parts.append(str(d))
        return " ".join(parts)

    results["PaddleOCR-VL"] = score("PaddleOCR-VL 0.9B", run_pvl)
except Exception as e:
    print(f"\n### PaddleOCR-VL НЕ ПОДНЯЛСЯ: {type(e).__name__}: {str(e)[:160]}")

# ---------- 2. DeepSeek-OCR ----------
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    MID = "deepseek-ai/DeepSeek-OCR"
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(MID, trust_remote_code=True,
                                    torch_dtype=torch.float32).eval()

    def run_dsocr(img, path):
        return mdl.infer(tok, prompt="<image>\n<|grounding|>OCR this image.",
                         image_file=path, output_path="/tmp/dsocr",
                         base_size=1024, image_size=640, crop_mode=True,
                         save_results=False, test_compress=False)

    results["DeepSeek-OCR"] = score("DeepSeek-OCR 3B", run_dsocr)
except Exception as e:
    print(f"\n### DeepSeek-OCR НЕ ПОДНЯЛСЯ: {type(e).__name__}: {str(e)[:160]}")

# ---------- 3. Qwen2.5-VL-3B ----------
try:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    MID = "Qwen/Qwen2.5-VL-3B-Instruct"
    proc = AutoProcessor.from_pretrained(MID)
    # device_map НЕ используем: он требует accelerate, а тот на этом сервере
    # не импортируется — тянет boto3, а системный pyOpenSSL сломан
    # (AttributeError: module 'lib' has no attribute 'GEN_EMAIL').
    qmdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.float32).eval()

    def run_qwen(img, path):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt")
        with torch.no_grad():
            out = qmdl.generate(**inp, max_new_tokens=256, do_sample=False)
        gen = out[0][inp["input_ids"].shape[1]:]
        return proc.decode(gen, skip_special_tokens=True)

    results["Qwen2.5-VL-3B"] = score("Qwen2.5-VL-3B", run_qwen)
except Exception as e:
    print(f"\n### Qwen2.5-VL НЕ ПОДНЯЛСЯ: {type(e).__name__}: {str(e)[:160]}")

print(f"\n\n{'='*60}\nСВОДКА (из {len(SAMPLES)} кропов, эталон DeepSeek API = 6/6)")
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"   {v}/{len(SAMPLES)}  {k}")
