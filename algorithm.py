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
    Hocanın slaytındaki varyant:
    1) EDD sırala
    2) soldan sağa ilerle; gecikme olursa o ANKİ işi çıkar ve rejected'a at
    """
    df = df.copy()
    _require_cols(df, ["DeliveryTime", "ProcessTime"], "Moore(Slide)")
    df = _ensure_job(df)
    df = _to_numeric(df, ["DeliveryTime", "ProcessTime"])
    df = df.dropna(subset=["DeliveryTime", "ProcessTime"]).reset_index(drop=True)

    # EDD
    df = df.sort_values(by=["DeliveryTime"]).reset_index(drop=True)

    schedule = []
    rejected = []
    t = 0.0

    for i in range(len(df)):
        job = df.loc[i].to_dict()
        schedule.append(job)
        t += float(job["ProcessTime"])

        if t > float(job["DeliveryTime"]):
            # SLAYT MANTIĞI: gecikmeye sebep olan (son eklenen) işi çıkar
            bad = schedule.pop(-1)
            rejected.append(bad)
            t -= float(bad["ProcessTime"])

    return (
        pd.DataFrame(schedule).reset_index(drop=True),
        pd.DataFrame(rejected).reset_index(drop=True),
        pd.concat([pd.DataFrame(schedule), pd.DataFrame(rejected)], ignore_index=True)
    )
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

def johnson_algorithm(df: pd.DataFrame):
    """
    Johnson Algoritması (2 Makine Flow Shop, M1 -> M2, amaç makespan Cmax min)

    Gerekli kolonlar:
      - Job (opsiyonel, yoksa otomatik)
      - M1Time (1. makine süreleri)
      - M2Time (2. makine süreleri)

    Dönen:
      - seq_df: Optimal iş sırası (Job, M1Time, M2Time)
      - schedule_df: Çözüm tablosu (M1 giriş/çıkış, M2 giriş/çıkış, M2 bekleme)
      - makespan: Cmax (son işin M2 çıkışı)
      - total_m2_wait: M2 toplam bekleme (atıl süre)
    """
    df = df.copy()
    _require_cols(df, ["M1Time", "M2Time"], "Johnson")
    df = _ensure_job(df)
    df = _to_numeric(df, ["M1Time", "M2Time"])
    df = df.dropna(subset=["M1Time", "M2Time"]).reset_index(drop=True)

    remaining = df.copy()
    left = []
    right = []

    # -------------------------
    # 1) Johnson sıralaması
    # -------------------------
    while len(remaining) > 0:
        m1 = remaining["M1Time"].astype(float).values
        m2 = remaining["M2Time"].astype(float).values
        mins = np.minimum(m1, m2)
        k = int(np.argmin(mins))
        row = remaining.iloc[k].to_dict()

        if float(row["M1Time"]) <= float(row["M2Time"]):
            left.append(row)
        else:
            right.append(row)

        remaining = remaining.drop(index=remaining.index[k]).reset_index(drop=True)

    sequence = left + right[::-1]
    seq_df = pd.DataFrame(sequence).reset_index(drop=True)

    # -------------------------
    # 2) Çözüm tablosu (doğru zaman akışı)
    # -------------------------
    m1_finish_prev = 0.0
    m2_finish_prev = 0.0

    rows = []
    total_m2_wait = 0.0

    for _, r in seq_df.iterrows():
        p1 = float(r["M1Time"])
        p2 = float(r["M2Time"])

        # M1
        m1_start = m1_finish_prev
        m1_finish = m1_start + p1

        # M2 (kritik kural)
        m2_start = max(m1_finish, m2_finish_prev)
        m2_finish = m2_start + p2

        # M2'nin atıl süresi = M2'nin önceki iş bitişi ile bu işin başlama arası
        m2_wait = max(0.0, m2_start - m2_finish_prev)
        total_m2_wait += m2_wait

        rows.append({
            "İş Sırası": r["Job"],
            "Pişirme (M1) Giriş": m1_start,
            "Pişirme (M1) Çıkış": m1_finish,
            "Süsleme (M2) Giriş": m2_start,
            "Süsleme (M2) Çıkış": m2_finish,
            "Atıl Süre (Bekleme)": m2_wait
        })

        # güncelle
        m1_finish_prev = m1_finish
        m2_finish_prev = m2_finish

    schedule_df = pd.DataFrame(rows)
    makespan = float(schedule_df["Süsleme (M2) Çıkış"].iloc[-1]) if not schedule_df.empty else float("inf")

    return seq_df, schedule_df, makespan, float(total_m2_wait)

def smith_algorithm(df: pd.DataFrame):
    """
    Smith Algoritması (notlardaki/slayttaki gibi):
    - Sağdan sola (K=n -> 1) yerleştirir
    - t = sum(Pi)
    - Her adımda Di >= t koşulunu sağlayan işler arasından Pi en büyük olan seçilir
    - Seçilen iş en sağdaki boş yere yazılır, t = t - Pi

    Gerekli kolonlar:
      - DeliveryTime (Di)
      - ProcessTime  (Pi)

    Dönen:
      - optimal_df: Smith sırası
      - rejected_df: Tmax=0 koşulu sağlanamazsa (eligible boş kalırsa) buraya düşer
                    (notlarda bu durum yok ama uygulama patlamasın diye ekledik)
    """
    df = df.copy()
    _require_cols(df, ["DeliveryTime", "ProcessTime"], "Smith")
    df = _ensure_job(df)
    df = _to_numeric(df, ["DeliveryTime", "ProcessTime"])
    df = df.dropna(subset=["DeliveryTime", "ProcessTime"]).reset_index(drop=True)

    remaining = df.copy()
    t = float(remaining["ProcessTime"].astype(float).sum())  # t = ΣPi

    placed_from_right = []   # K=n..1 seçtiklerimiz (sağdan sola)
    rejected = []

    while len(remaining) > 0:
        due = remaining["DeliveryTime"].astype(float)
        p = remaining["ProcessTime"].astype(float)

        eligible = remaining[due >= t]

        if not eligible.empty:
            # Koşul (b): eligible içinden Pi max seç
            p_el = eligible["ProcessTime"].astype(float)
            max_p = float(p_el.max())
            cand = eligible[p_el == max_p]

            # Tie-break (tamamen stabilite için): due büyük olanı seçelim
            if len(cand) > 1:
                idx = cand["DeliveryTime"].astype(float).idxmax()
            else:
                idx = cand.index[0]
        else:
            # Tmax=0 mümkün değil -> uygulama kırılmasın
            # not: tezde Tmax=0 koşulu sağlandığı varsayılıyor
            idx = remaining["DeliveryTime"].astype(float).idxmax()
            rejected.append(remaining.loc[idx].to_dict())

        job = remaining.loc[idx].to_dict()
        placed_from_right.append(job)

        t -= float(job["ProcessTime"])
        remaining = remaining.drop(index=idx).reset_index(drop=True)

    # sağdan sola yerleştirdiklerimizi ters çevir -> soldan sağa final sıra
    optimal_df = pd.DataFrame(list(reversed(placed_from_right))).reset_index(drop=True)
    rejected_df = pd.DataFrame(rejected).reset_index(drop=True)

    return optimal_df, rejected_df