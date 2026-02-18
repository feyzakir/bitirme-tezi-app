import pandas as pd
import numpy as np
# ----------------------------
# Helpers
# ----------------------------
def _require_cols(df: pd.DataFrame, cols: list[str], algo: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{algo} için eksik kolon(lar): {missing}")

def _ensure_job(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Job" not in df.columns:
        df.insert(0, "Job", range(1, len(df) + 1))
    df["Job"] = pd.to_numeric(df["Job"], errors="coerce")
    df["Job"] = df["Job"].fillna(pd.Series(range(1, len(df) + 1))).astype(int)
    return df

def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =========================================================
# Moore-Hodgson (1||ΣUj)
# =========================================================
def moore_algorithm(df: pd.DataFrame):
    """
    1) EDD sırala
    2) kümülatif süre due'yu aşarsa schedule içinden en büyük p'yi çıkar (rejected)
    """
    df = df.copy()
    _require_cols(df, ["DeliveryTime", "ProcessTime"], "Moore")
    df = _ensure_job(df)
    df = _to_numeric(df, ["DeliveryTime", "ProcessTime"])
    df = df.dropna(subset=["DeliveryTime", "ProcessTime"]).reset_index(drop=True)
    df = df.sort_values(by="DeliveryTime").reset_index(drop=True)

    schedule = []   # list[dict]
    rejected = []   # list[dict]
    t = 0.0
    for i in range(len(df)):
        job = df.loc[i].to_dict()
        schedule.append(job)
        t += float(job["ProcessTime"])

        if t > float(job["DeliveryTime"]):
            # ✅ idxmax/pop bug'ını bitiren kısım:
            p_list = [float(x["ProcessTime"]) for x in schedule]
            pos = int(np.argmax(p_list))           # liste pozisyonu
            rejected.append(schedule[pos])
            t -= float(schedule[pos]["ProcessTime"])
            schedule.pop(pos)

    return pd.DataFrame(schedule).reset_index(drop=True), pd.DataFrame(rejected).reset_index(drop=True)

# =========================================================
# Basit tek-makine sıralamalar
# =========================================================
def spt_algorithm(df: pd.DataFrame):
    df = df.copy()
    _require_cols(df, ["ProcessTime"], "SPT")
    df = _ensure_job(df)
    df = _to_numeric(df, ["ProcessTime"])
    df = df.dropna(subset=["ProcessTime"]).reset_index(drop=True)
    return df.sort_values(by="ProcessTime").reset_index(drop=True), pd.DataFrame()
def edd_algorithm(df: pd.DataFrame):
    df = df.copy()
    _require_cols(df, ["DeliveryTime"], "EDD")
    df = _ensure_job(df)
    df = _to_numeric(df, ["DeliveryTime"])
    df = df.dropna(subset=["DeliveryTime"]).reset_index(drop=True)
    return df.sort_values(by="DeliveryTime").reset_index(drop=True), pd.DataFrame()
def lpt_algorithm(df: pd.DataFrame):
    df = df.copy()
    _require_cols(df, ["ProcessTime"], "LPT")
    df = _ensure_job(df)
    df = _to_numeric(df, ["ProcessTime"])
    df = df.dropna(subset=["ProcessTime"]).reset_index(drop=True)
    return df.sort_values(by="ProcessTime", ascending=False).reset_index(drop=True), pd.DataFrame()
def fcfs_algorithm(df: pd.DataFrame):
    df = _ensure_job(df)
    return df.reset_index(drop=True), pd.DataFrame()
def lifo_algorithm(df: pd.DataFrame):
    """
    LIFO (Last In First Out): Son giren ilk çıkar.
    Burada 'giriş sırası' = tablodaki mevcut satır sırası (index sırası).
    Dolayısıyla DataFrame'i tersten çeviriyoruz.
    """
    df = _ensure_job(df)
    return df.iloc[::-1].reset_index(drop=True), pd.DataFrame()

# =========================================================
# CR (Critical Ratio)
# =========================================================
def cr_algorithm(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)
    _require_cols(df, ["DeliveryTime", "ProcessTime"], "CR")
    df = _ensure_job(df)
    df = _to_numeric(df, ["DeliveryTime", "ProcessTime"])
    df = df.dropna(subset=["DeliveryTime", "ProcessTime"]).reset_index(drop=True)
    remaining = df.copy()
    seq = []
    t = 0.0
    while len(remaining) > 0:
        p = remaining["ProcessTime"].replace(0, np.nan)
        cr = (remaining["DeliveryTime"] - t) / p
        cr = cr.fillna(np.inf)

        idx = cr.idxmin()
        job = remaining.loc[idx].to_dict()
        seq.append(job)

        t += float(job["ProcessTime"])
        remaining = remaining.drop(index=idx)

    return pd.DataFrame(seq).reset_index(drop=True), pd.DataFrame()
# =========================================================
# MDD (Modified Due Date)
# =========================================================
def mdd_algorithm(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)
    _require_cols(df, ["DeliveryTime", "ProcessTime"], "MDD")
    df = _ensure_job(df)
    df = _to_numeric(df, ["DeliveryTime", "ProcessTime"])
    df = df.dropna(subset=["DeliveryTime", "ProcessTime"]).reset_index(drop=True)
    remaining = df.copy()
    seq = []
    t = 0.0
    while len(remaining) > 0:
        p = remaining["ProcessTime"].astype(float)
        due = remaining["DeliveryTime"].astype(float)
        pr = np.maximum(due.values, (t + p).values)
        idx = remaining.index[int(np.argmin(pr))]
        job = remaining.loc[idx].to_dict()
        seq.append(job)
        t += float(job["ProcessTime"])
        remaining = remaining.drop(index=idx)
    return pd.DataFrame(seq).reset_index(drop=True), pd.DataFrame()

