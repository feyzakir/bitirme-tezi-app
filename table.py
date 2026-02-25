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
    Normalize / temizle:
    - Tek makine: Job, ProcessTime, DeliveryTime
    - Johnson (2 makine): Job, M1Time, M2Time
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 1) Kolon isimlerini normalize ederek rename haritası oluştur
    rename = {}
    for c in df.columns:
        k = _normalize_header(c)

        # Job
        if k in {"job", "is", "isno", "jobid", "id"}:
            rename[c] = "Job"

        # Tek makine
        elif k in {"processtime", "ptime", "sure", "islemsuresi"}:
            rename[c] = "ProcessTime"
        elif k in {"deliverytime", "duedate", "due", "teslimtarihi", "teslim"}:
            rename[c] = "DeliveryTime"

        # Johnson 2 makine: M1/M2
        elif k in {"m1time", "m1", "makine1", "machine1", "p", "pisirme", "firin"}:
            rename[c] = "M1Time"
        elif k in {"m2time", "m2", "makine2", "machine2", "s", "susleme", "dekor"}:
            rename[c] = "M2Time"

    if rename:
        df = df.rename(columns=rename)

    # 2) Johnson mı, tek makine mi?
    has_single = ("ProcessTime" in df.columns) and ("DeliveryTime" in df.columns)
    has_johnson = ("M1Time" in df.columns) and ("M2Time" in df.columns)

    if not has_single and not has_johnson:
        return pd.DataFrame()

    # 3) Job yoksa ekle (string olsun; A,B,C gibi değerleri bozmayalım)
    if "Job" not in df.columns:
        df.insert(0, "Job", list(range(1, len(df) + 1)))

    # Job'u metin gibi sakla (Johnson'da A,B,C var)
    df["Job"] = df["Job"].astype(str).str.strip()
    df.loc[df["Job"].isin(["", "nan", "None"]), "Job"] = None

    # boş Job'lara otomatik numara bas
    if df["Job"].isna().any():
        fill_vals = [str(i) for i in range(1, len(df) + 1)]
        df.loc[df["Job"].isna(), "Job"] = pd.Series(fill_vals)[df["Job"].isna()].values

    # 4) Modlara göre numeric + dropna
    if has_johnson and not has_single:
        # Johnson modu
        df["M1Time"] = pd.to_numeric(df["M1Time"], errors="coerce")
        df["M2Time"] = pd.to_numeric(df["M2Time"], errors="coerce")
        df = df.dropna(subset=["M1Time", "M2Time"]).reset_index(drop=True)
        return df[["Job", "M1Time", "M2Time"]].reset_index(drop=True)

    if has_single and not has_johnson:
        # Tek makine modu
        df["ProcessTime"] = pd.to_numeric(df["ProcessTime"], errors="coerce")
        df["DeliveryTime"] = pd.to_numeric(df["DeliveryTime"], errors="coerce")
        df = df.dropna(subset=["ProcessTime", "DeliveryTime"]).reset_index(drop=True)
        return df[["Job", "DeliveryTime", "ProcessTime"]].reset_index(drop=True)

    # 5) İkisi birden varsa: kullanıcıya sürpriz olmasın diye TEK MAKİNE döndür
    # (İstersen burada Johnson'ı tercih ettirebiliriz.)
    df["ProcessTime"] = pd.to_numeric(df["ProcessTime"], errors="coerce")
    df["DeliveryTime"] = pd.to_numeric(df["DeliveryTime"], errors="coerce")
    df = df.dropna(subset=["ProcessTime", "DeliveryTime"]).reset_index(drop=True)
    return df[["Job", "DeliveryTime", "ProcessTime"]].reset_index(drop=True)


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
