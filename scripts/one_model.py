"""Тест ОДНОЙ модели в отдельном процессе, с замером RSS по шагам.

Запуск: python3 scripts/one_model.py qwen|dsocr|pvl
Печатает память после каждого шага, чтобы при SIGKILL было видно,
на какой отметке процесс убили.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.ERROR)

WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen"
IMG = "data/images/1300656060_674671661_23_1780483810_3.jpg"   # эталон Андрея
EXPECT = "+79859791299"


def rss():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS"):
                    return ln.split()[1] + " КБ"
    except Exception:
        pass
    return "?"


def step(msg):
    print(f"[{rss():>12}] {msg}", flush=True)


step("старт")
from PIL import Image
img = Image.open(IMG).convert("RGB")
step(f"картинка загружена {img.size}")

text = ""
t0 = time.monotonic()

if WHICH == "pvl":
    from paddleocr import PaddleOCRVL
    step("paddleocr импортирован")
    p = PaddleOCRVL()
    step("модель поднята")
    out = p.predict(IMG)
    step("инференс выполнен")
    text = " ".join(str(r if isinstance(r, dict) else (getattr(r, "json", None) or {}))
                    for r in out)

elif WHICH == "dsocr":
    import torch
    from transformers import AutoModel, AutoTokenizer
    step("transformers импортирован")
    MID = "deepseek-ai/DeepSeek-OCR"
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    step("токенизатор загружен")
    mdl = AutoModel.from_pretrained(MID, trust_remote_code=True,
                                    torch_dtype=torch.float32).eval()
    step("веса загружены")
    text = str(mdl.infer(tok, prompt="<image>\nFree OCR.", image_file=IMG,
                         output_path="/tmp/ds", base_size=640, image_size=640,
                         crop_mode=False, save_results=False, test_compress=False))
    step("инференс выполнен")

elif WHICH == "qwen":
    import torch
    torch.set_num_threads(8)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    step("transformers импортирован")
    MID = "Qwen/Qwen2.5-VL-3B-Instruct"
    proc = AutoProcessor.from_pretrained(MID)
    step("процессор загружен")
    # bfloat16 вместо float32: 3B в fp32 это 12 ГБ, вдвое меньше даёт шанс
    # уложиться в лимит шейреда. device_map не берём — он требует accelerate,
    # который на этом сервере ломает импорт OWLv2 и весь парсер.
    mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    step("веса загружены")
    img.thumbnail((768, 768))
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Извлеки весь текст. Точно перепиши цифры телефона."}]}]
    t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[t], images=[img], return_tensors="pt")
    step("вход подготовлен")
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=128, do_sample=False)
    step("инференс выполнен")
    text = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

print(f"\nвремя инференса: {time.monotonic() - t0:.1f} с", flush=True)
print("ВЫВОД:", repr(text[:400]), flush=True)
from banner_parser.ocr import extract_contacts
ph = extract_contacts(text).phones
print("ТЕЛЕФОНЫ:", ph, "| ожидался", EXPECT, flush=True)
print("РЕЗУЛЬТАТ:", "ВЕРНО" if EXPECT in str(ph) else "промах", flush=True)
