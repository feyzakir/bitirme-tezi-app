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
    """
    Job sütununu kullanıcı nasıl girdiyse o formatta KORU:
    - Hepsi sayısal ise int olarak tut (1,2,3 gibi)
    - İçinde harf/karışık varsa string olarak tut (A,B,C gibi)
    - Job yoksa 1..n üret
    - Boş Job hücrelerini uygun tipte doldur
    """
    df = df.copy()

    # Job yoksa 1..n oluştur
    if "Job" not in df.columns:
        df.insert(0, "Job", list(range(1, len(df) + 1)))
        return df

    s = df["Job"]

    # Tümü sayıya çevrilebiliyorsa sayısal kabul et
    num = pd.to_numeric(s, errors="coerce")
    all_numeric = num.notna().all()

    if all_numeric:
        # Sayısal Job: int yap, boş varsa sıra numarasıyla doldur
        df["Job"] = num.astype(int)
        if df["Job"].isna().any():
            # çok nadir: all_numeric True iken NaN olmaz ama güvenlik
            fill_vals = pd.Series(range(1, len(df) + 1), index=df.index)
            df["Job"] = df["Job"].fillna(fill_vals).astype(int)
    else:
        # Harf/karışık Job: string olarak koru, boşları "1","2" diye doldur
        df["Job"] = s.astype(str)
        mask_empty = df["Job"].isna() | df["Job"].str.strip().isin(["", "nan", "None"])
        if mask_empty.any():
            fill_vals = pd.Series([str(i) for i in range(1, len(df) + 1)], index=df.index)
            df.loc[mask_empty, "Job"] = fill_vals[mask_empty]

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
      - Machine1 Process Time
      - Machine2 Process Time
    """
    df = df.copy()
    _require_cols(df, ["Machine1 Process Time", "Machine2 Process Time"], "Johnson")
    df = _ensure_job(df)
    df = _to_numeric(df, ["Machine1 Process Time", "Machine2 Process Time"])
    df = df.dropna(subset=["Machine1 Process Time", "Machine2 Process Time"]).reset_index(drop=True)

    # -------------------------
    # 1) Johnson sıralaması
    # -------------------------
    remaining = df.copy()
    left = []
    right = []

    while len(remaining) > 0:
        m1 = remaining["Machine1 Process Time"].astype(float).values
        m2 = remaining["Machine2 Process Time"].astype(float).values

        mins = np.minimum(m1, m2)
        k = int(np.argmin(mins))
        row = remaining.iloc[k].to_dict()

        if float(row["Machine1 Process Time"]) <= float(row["Machine2 Process Time"]):
            left.append(row)
        else:
            right.append(row)

        remaining = remaining.drop(index=remaining.index[k]).reset_index(drop=True)

    sequence = left + right[::-1]
    seq_df = pd.DataFrame(sequence)[
        ["Job", "Machine1 Process Time", "Machine2 Process Time"]
    ].reset_index(drop=True)

    # -------------------------
    # 2) Çözüm tablosu (zaman akışı)
    # -------------------------
    m1_finish_prev = 0.0
    m2_finish_prev = 0.0

    rows = []
    total_m2_wait = 0.0

    for _, r in seq_df.iterrows():
        p1 = float(r["Machine1 Process Time"])
        p2 = float(r["Machine2 Process Time"])

        # M1
        m1_start = m1_finish_prev
        m1_finish = m1_start + p1

        # M2
        m2_start = max(m1_finish, m2_finish_prev)
        m2_finish = m2_start + p2

        # M2 idle
        m2_wait = max(0.0, m2_start - m2_finish_prev)
        total_m2_wait += m2_wait

        rows.append({
            "İş Sırası": r["Job"],
            "M1 Başla": m1_start,
            "M1 Bitir": m1_finish,
            "M2 Başla": m2_start,
            "M2 Bitir": m2_finish,
            "M2 Atıl Süre": m2_wait
        })

        m1_finish_prev = m1_finish
        m2_finish_prev = m2_finish

    schedule_df = pd.DataFrame(rows)
    makespan = float(schedule_df["M2 Bitir"].iloc[-1]) if not schedule_df.empty else float("inf")

    return seq_df, schedule_df, makespan, float(total_m2_wait)
def johnson_3machine_algorithm(df: pd.DataFrame):
    """
    Johnson Algoritması - 3 Makine (Flow Shop: M1 -> M2 -> M3) | Hoca notu varyantı

    Koşul (en az biri sağlanmalı):
      1) min(P1) >= max(P2)  veya
      2) min(P3) >= max(P2)

    İndirgeme:
      M1' = P1 + P2
      M2' = P2 + P3

    CT hesabı (slayttaki gibi):
      CT(M1') = kümülatif sum(M1')
      CT(M2') = M1'(ilk iş) + kümülatif sum(M2')
      Cmax = max(CT(M1')_son, CT(M2')_son)

    Dönen:
      - seq_df: optimal sıra (Job + P1,P2,P3 + M1',M2')
      - schedule_df: slayt formatı CT tablosu
      - makespan: Cmax (slayttaki CT mantığı)
      - idle_m2: (bu varyantta hesaplanmıyor -> 0)
      - idle_m3: (bu varyantta hesaplanmıyor -> 0)
    """
    df = df.copy()
    _require_cols(df, ["Machine1 Process Time", "Machine2 Process Time", "Machine3 Process Time"], "Johnson-3M")
    df = _ensure_job(df)
    df = _to_numeric(df, ["Machine1 Process Time", "Machine2 Process Time", "Machine3 Process Time"])
    df = df.dropna(subset=["Machine1 Process Time", "Machine2 Process Time", "Machine3 Process Time"]).reset_index(drop=True)

    # 1) Koşul kontrolü
    p1min = float(df["Machine1 Process Time"].astype(float).min())
    p2max = float(df["Machine2 Process Time"].astype(float).max())
    p3min = float(df["Machine3 Process Time"].astype(float).min())

    cond1 = p1min >= p2max
    cond2 = p3min >= p2max
    if not (cond1 or cond2):
        raise ValueError(
            f"Johnson (3 makine) uygulanamaz: "
            f"min(P1)={p1min:.2f} < max(P2)={p2max:.2f} ve "
            f"min(P3)={p3min:.2f} < max(P2)={p2max:.2f}."
        )

    # 2) Dummy makineler
    reduced = df.copy()
    reduced["M1'"] = reduced["Machine1 Process Time"].astype(float) + reduced["Machine2 Process Time"].astype(float)
    reduced["M2'"] = reduced["Machine2 Process Time"].astype(float) + reduced["Machine3 Process Time"].astype(float)

    # 3) Johnson sıralaması (M1', M2' üzerinden)
    remaining = reduced.copy()
    left, right = [], []

    while len(remaining) > 0:
        m1p = remaining["M1'"].astype(float).values
        m2p = remaining["M2'"].astype(float).values
        mins = np.minimum(m1p, m2p)
        k = int(np.argmin(mins))
        row = remaining.iloc[k].to_dict()

        if float(row["M1'"]) <= float(row["M2'"]):
            left.append(row)
        else:
            right.append(row)

        remaining = remaining.drop(index=remaining.index[k]).reset_index(drop=True)

    sequence = left + right[::-1]
    seq_df = pd.DataFrame(sequence).reset_index(drop=True)

    # 4) Slayt CT hesabı
    m1p_list = seq_df["M1'"].astype(float).tolist()
    m2p_list = seq_df["M2'"].astype(float).tolist()

    ct_m1p = np.cumsum(m1p_list)  # CT(M1') = cumulative sum
    first_m1p = float(m1p_list[0]) if len(m1p_list) > 0 else 0.0
    ct_m2p = first_m1p + np.cumsum(m2p_list)  # CT(M2') = first M1' + cumulative sum(M2')

    makespan = float(max(ct_m1p[-1], ct_m2p[-1])) if len(m1p_list) > 0 else float("inf")

    # 5) Slayt tarzı tablo (schedule_df)
    schedule_df = pd.DataFrame({
        "İş Sırası": seq_df["Job"],
        "M1' (P1+P2)": seq_df["M1'"].astype(float),
        "M2' (P2+P3)": seq_df["M2'"].astype(float),
        "CT M1'": ct_m1p,
        "CT M2'": ct_m2p,
    })

    # Bu varyantta M2/M3 idle slaytta bu CT hesabının parçası değil.
    idle_m2 = 0.0
    idle_m3 = 0.0

    return seq_df, schedule_df, makespan, idle_m2, idle_m3
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

# =========================================================
# Workstation Sequencing (Priority Rules) - NEW
# Columns:
#   - TimeSinceOrderReceived
#   - ProcessTime
#   - TimeRemainingUntilDeliveryDate
# Rules: FIFO, EDD, SPT, LPT, LIFO, SRIT
# Metrics: Avg Delay, Avg Lead Time, #Delayed Jobs
# =========================================================

def _require_ws_cols(df: pd.DataFrame):
    _require_cols(df, ["ProcessTime", "TimeSinceOrderReceived", "TimeRemainingUntilDeliveryDate"], "Workstation")

def _ws_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_job(df)  # Job A/B/C ya da 1/2/3 korunur
    _require_ws_cols(df)
    df = _to_numeric(df, ["ProcessTime", "TimeSinceOrderReceived", "TimeRemainingUntilDeliveryDate"])
    df = df.dropna(subset=["ProcessTime", "TimeSinceOrderReceived", "TimeRemainingUntilDeliveryDate"]).reset_index(drop=True)

    # Remaining idle time (slayttaki B-A): RemainingDue - ProcessTime
    df["RemainingIdleTime"] = df["TimeRemainingUntilDeliveryDate"].astype(float) - df["ProcessTime"].astype(float)
    return df

def workstation_schedule_and_metrics(df_sorted: pd.DataFrame):
    """
    Slayttaki tablo mantığı:
      Start = önceki Finish
      Finish = Start + ProcessTime
      Delay = max(0, Finish - TimeRemainingUntilDeliveryDate)
      LeadTime = TimeSinceOrderReceived + Finish
    """
    df_sorted = df_sorted.reset_index(drop=True).copy()

    t = 0.0
    rows = []
    for _, r in df_sorted.iterrows():
        p = float(r["ProcessTime"])
        start = t
        finish = start + p
        delay = max(0.0, finish - float(r["TimeRemainingUntilDeliveryDate"]))
        lead = float(r["TimeSinceOrderReceived"]) + finish

        rows.append({
            "Order": r["Job"],
            "Time since order received (days)": float(r["TimeSinceOrderReceived"]),
            "Process (Cycle) time (days)": float(r["ProcessTime"]),
            "Time remaining until delivery date (days)": float(r["TimeRemainingUntilDeliveryDate"]),
            "Remaining idle time (days)": float(r["RemainingIdleTime"]),
            "Start time (day)": start,
            "Finish time (day)": finish,
            "Delay time (day)": delay,
            "Lead time (day)": lead,
        })
        t = finish

    schedule_df = pd.DataFrame(rows)

    avg_delay = float(schedule_df["Delay time (day)"].mean()) if not schedule_df.empty else float("inf")
    avg_lead = float(schedule_df["Lead time (day)"].mean()) if not schedule_df.empty else float("inf")
    delayed_jobs = int((schedule_df["Delay time (day)"] > 0).sum()) if not schedule_df.empty else 0

    return schedule_df, avg_delay, avg_lead, delayed_jobs

# ---- Priority rules (6) ----
def ws_fifo(df: pd.DataFrame):
    df = _ws_prepare(df)
    # FIFO: oldest -> newest (time since received büyük olan önce)
    seq = df.sort_values(by="TimeSinceOrderReceived", ascending=False)
    return workstation_schedule_and_metrics(seq)

def ws_edd(df: pd.DataFrame):
    df = _ws_prepare(df)
    # EDD: earliest delivery date first => remaining time küçük olan önce
    seq = df.sort_values(by="TimeRemainingUntilDeliveryDate", ascending=True)
    return workstation_schedule_and_metrics(seq)

def ws_spt(df: pd.DataFrame):
    df = _ws_prepare(df)
    seq = df.sort_values(by="ProcessTime", ascending=True)
    return workstation_schedule_and_metrics(seq)

def ws_lpt(df: pd.DataFrame):
    df = _ws_prepare(df)
    seq = df.sort_values(by="ProcessTime", ascending=False)
    return workstation_schedule_and_metrics(seq)

def ws_lifo(df: pd.DataFrame):
    df = _ws_prepare(df)
    # LIFO: newest -> oldest (time since received küçük olan önce)
    seq = df.sort_values(by="TimeSinceOrderReceived", ascending=True)
    return workstation_schedule_and_metrics(seq)

def ws_srit(df: pd.DataFrame):
    df = _ws_prepare(df)
    # SRIT: remaining idle time (B-A) en küçük olan önce
    seq = df.sort_values(by="RemainingIdleTime", ascending=True)
    return workstation_schedule_and_metrics(seq)