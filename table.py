import re
import io
import pandas as pd
from PIL import Image

# =========================================================
# 1) Helpers
# =========================================================
def _normalize_header(s: str) -> str:
    s = str(s).strip().lower()
    s = (s.replace("ı", "i").replace("ş", "s").replace("ğ", "g")
           .replace("ü", "u").replace("ö", "o").replace("ç", "c"))
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s

def _finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tek-makine için normalize:
    - Kolon isimlerini eşleştir (Job, ProcessTime, DeliveryTime)
    - Sayısala çevir
    - Boş/NaN satırları temizle
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        k = _normalize_header(c)
        if k in {"job", "is", "isno", "jobid", "id"}:
            rename[c] = "Job"
        elif k in {"processtime", "ptime", "sure", "islemsuresi", "p"}:
            rename[c] = "ProcessTime"
        elif k in {"deliverytime", "duedate", "due", "teslimtarihi", "teslim", "d"}:
            rename[c] = "DeliveryTime"

    if rename:
        df = df.rename(columns=rename)

    # minimum kolon
    if "ProcessTime" not in df.columns or "DeliveryTime" not in df.columns:
        return pd.DataFrame()

    if "Job" not in df.columns:
        df.insert(0, "Job", list(range(1, len(df) + 1)))

    # sayısallaştır
    df["ProcessTime"] = pd.to_numeric(df["ProcessTime"], errors="coerce")
    df["DeliveryTime"] = pd.to_numeric(df["DeliveryTime"], errors="coerce")

    # zorunlu kolonlar boş olmasın
    df = df.dropna(subset=["ProcessTime", "DeliveryTime"]).reset_index(drop=True)

    # Job integer yap
    df["Job"] = pd.to_numeric(df["Job"], errors="coerce")
    df["Job"] = df["Job"].fillna(pd.Series(range(1, len(df) + 1))).astype(int)

    return df
def finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    return _finalize_df(df)
# =========================================================
# 2) OCR Engines
# =========================================================
_HY_READY = False
_HY_PROC = None
_HY_MODEL = None

def _try_load_hunyuan():
    global _HY_READY, _HY_PROC, _HY_MODEL
    if _HY_READY:
        return True

    try:
        import torch
        from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

        if not torch.cuda.is_available():
            return False

        model_name = "tencent/HunyuanOCR"
        proc = AutoProcessor.from_pretrained(model_name, use_fast=False)

        model = HunYuanVLForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="eager",
            dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

        _HY_PROC, _HY_MODEL = proc, model
        _HY_READY = True
        return True

    except Exception:
        return False

def _hunyuan_ocr_to_text(image: Image.Image) -> str:
    import torch
    proc, model = _HY_PROC, _HY_MODEL

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text":
                "Read the table and output ONLY CSV.\n"
                "Columns: Job, ProcessTime, DeliveryTime.\n"
                "No explanations."
            }
        ]}
    ]
    prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[prompt], images=[image], padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    in_len = inputs["input_ids"].shape[1]
    gen = out_ids[:, in_len:]
    out_text = proc.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return out_text
_PAD = None
def _get_paddle():
    global _PAD
    if _PAD is None:
        from paddleocr import PaddleOCR
        _PAD = PaddleOCR(use_angle_cls=True, lang="en")
    return _PAD

def _paddle_read_lines(image: Image.Image) -> list[str]:
    import cv2
    import numpy as np
    ocr = _get_paddle()
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res = ocr.ocr(img, cls=True)
    lines = []
    if res and len(res) > 0:
        for block in res[0]:
            txt = block[1][0]
            if txt:
                lines.append(txt)
    return lines
def _lines_to_table(lines: list[str]) -> pd.DataFrame:
    rows = []
    for ln in lines:
        t = ln.strip()
        nums = re.findall(r"\d+(?:[.,]\d+)?", t)
        if len(nums) >= 2:
            left = re.split(r"\d", t, maxsplit=1)[0].strip()
            p = nums[0].replace(",", ".")
            d = nums[1].replace(",", ".")
            job = left if left else str(len(rows) + 1)
            rows.append([job, p, d])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["Job", "ProcessTime", "DeliveryTime"])
# =========================================================
# 3) Main API: extract_jobs_from_image
# =========================================================
def extract_jobs_from_image(uploaded_file) -> pd.DataFrame:
    image = Image.open(uploaded_file).convert("RGB")

    # 1) GPU varsa HunyuanOCR
    if _try_load_hunyuan():
        try:
            text = _hunyuan_ocr_to_text(image)

            t = text.strip()
            # kod fence temizle
            t = t.strip("`").strip()

            first = next((ln for ln in t.splitlines() if ln.strip()), "")
            sep = "," if first.count(",") >= first.count(";") else ";"

            df = pd.read_csv(io.StringIO(t), sep=sep)
            return _finalize_df(df)

        except Exception:
            pass

    # 2) CPU PaddleOCR fallback
    lines = _paddle_read_lines(image)
    df = _lines_to_table(lines)
    return _finalize_df(df)
