import streamlit as st
import pandas as pd
import io
import numpy as np

from table import extract_jobs_from_image, finalize_df
from algorithm import (
    moore_algorithm, spt_algorithm, edd_algorithm, lpt_algorithm, fcfs_algorithm,lifo_algorithm, cr_algorithm, mdd_algorithm, johnson_algorithm, johnson_3machine_algorithm, smith_algorithm,ws_fifo, ws_edd, ws_spt, ws_lpt, ws_lifo, ws_srit, lawler_algorithm
)

def total_completion_time(single_machine_df: pd.DataFrame) -> float:
    if single_machine_df is None or single_machine_df.empty:
        return float("inf")
    p = pd.to_numeric(single_machine_df["ProcessTime"], errors="coerce").fillna(0).astype(float).values
    C = p.cumsum()
    return float(C.sum())

def makespan(single_machine_df: pd.DataFrame) -> float:
    """Cmax = son bitiş zamanı = ΣP"""
    if single_machine_df is None or single_machine_df.empty:
        return float("inf")
    p = pd.to_numeric(single_machine_df["ProcessTime"], errors="coerce").fillna(0).astype(float).values
    return float(p.sum())


st.set_page_config(page_title="İş Sıralama Optimizasyonu", layout="centered")
st.title("İş Sıralama ve Üretim Planlama Aracı")
def render_comparison_panel():
    st.markdown("---")
    st.subheader("📊 Karşılaştırma Paneli (Kaydedilen Sonuçlar)")

    colx, coly = st.columns([1, 1])
    with colx:
        show_cmp = st.button("📊 Karşılaştırmayı Göster")
    with coly:
        clear_cmp = st.button("🧹 Kayıtlı Sonuçları Temizle")

    if clear_cmp:
        st.session_state.results = {}
        st.success("✅ Kayıtlı sonuçlar temizlendi.")

    if show_cmp:
        if not st.session_state.results:
            st.info("Henüz kaydedilmiş sonuç yok. Önce bir algoritma çalıştırıp kaydedin.")
        else:
            rows = []
            for algo_code, info in st.session_state.results.items():
                rows.append({
                    "Algoritma": info["label"],
                    "Kod": algo_code,
                    "Toplam Completion (∑Ci)": float(info["sumC"])
                })

            summary = pd.DataFrame(rows).sort_values("Toplam Completion (∑Ci)").reset_index(drop=True)

            summary["Toplam Completion (∑Ci)"] = summary["Toplam Completion (∑Ci)"].apply(
                lambda x: int(x) if float(x).is_integer() else round(x, 2)
            )

            best_value = summary["Toplam Completion (∑Ci)"].min()

            def highlight_best(row):
                return ["font-weight: bold"] * len(row) if row["Toplam Completion (∑Ci)"] == best_value else [""] * len(row)

            st.dataframe(summary.style.apply(highlight_best, axis=1), use_container_width=True)

            best_code = summary.loc[0, "Kod"]
            best = st.session_state.results[best_code]

            st.success(f"🏆 En iyi (en düşük ∑Ci): {best['label']}")
            st.subheader("✅ En iyi algoritmanın iş sırası sonucu")
            st.dataframe(best["df"], use_container_width=True)
            best_code = summary.loc[0, "Kod"]
            best = st.session_state.results[best_code]

            best_label = best["label"]
            best_score = float(best["sumC"])
            best_df = best["df"]

            excel_bytes = build_excel_bytes_best_only(
                best_label=best_label,
                best_code=best_code,
                best_score=best_score,
                best_df=best_df
)

            st.download_button(
                label="📥 Excel Olarak İndir",
                data=excel_bytes,
                file_name="sonuclar.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ==========================
# Excel Çıktı Oluştur
# ==========================
import io
import re

def safe_sheet_name(name: str) -> str:
    # Excel sheet adı max 31 karakter + bazı karakterler yasak
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name))
    return name[:31] if len(name) > 31 else name


import io

def build_excel_bytes_best_only(best_label: str, best_code: str, best_score: float, best_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()

    # openpyxl yoksa xlsxwriter'a düş
    try:
        engine = "openpyxl"
        import openpyxl  # noqa: F401
    except Exception:
        engine = "xlsxwriter"

    with pd.ExcelWriter(output, engine=engine) as writer:
        # 1) Üstte tek satırlık özet
        header_df = pd.DataFrame([{
            "En iyi algoritma": best_label,
            "Kod": best_code,
            "Completion (∑Ci)": best_score
        }])
        header_df.to_excel(writer, sheet_name="EnIyi", index=False, startrow=0)

        # 2) Altta optimum sıralama tablosu (2 satır boşluk bırakalım)
        start_row = len(header_df) + 3
        best_df.to_excel(writer, sheet_name="EnIyi", index=False, startrow=start_row)

    output.seek(0)
    return output.getvalue()
# ----------------------------
# Session State
# ----------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "manual_df" not in st.session_state:
    st.session_state.manual_df = None

if "results" not in st.session_state:
    st.session_state.results = {}  # {algo_code: {"label": str, "df": DataFrame, "sumC": float}}



input_method = st.radio("Veri giriş yöntemini seçin:", ["📷 Görsel Yükle", "📝 Manuel Giriş"])


# =========================================================
# 1) OCR GİRİŞİ (Görsel yükle -> tabloyu düzenle -> kaydet)
# =========================================================
if input_method == "📷 Görsel Yükle":
    uploaded_file = st.file_uploader("Tablonun bulunduğu görseli yükleyin:", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Yüklenen Görsel", use_container_width=True)

        df_ocr = extract_jobs_from_image(uploaded_file)

        if df_ocr is None or df_ocr.empty:
            st.warning("❌ Görselden geçerli bir tablo okunamadı.")
        else:
            st.success("✅ OCR tablo okundu. Aşağıda düzenleyebilirsiniz (satır ekle/sil serbest).")

            edited_ocr = st.data_editor(
                df_ocr,
                num_rows="dynamic",            # ✅ sınırsız satır
                use_container_width=True,
                key="editor_ocr"
            )

            if st.button("✅ OCR Tablosunu Kaydet ve Kullan"):
                fixed = finalize_df(edited_ocr)
                if fixed.empty:
                    st.error("⚠️ Tablo geçersiz. DeliveryTime ve ProcessTime dolu olmalı.")
                else:
                    st.session_state.df = fixed
                    st.success("✅ OCR tablo kaydedildi ve algoritmalara hazır.")
                    st.dataframe(fixed, use_container_width=True)


# =========================================================
# 2) MANUEL GİRİŞ (Excel gibi tablo oluştur/düzenle -> kaydet)
# =========================================================
elif input_method == "📝 Manuel Giriş":
    st.markdown("### 🧾 Excel Formatında Tablo Oluştur / Düzenle")

    table_mode = st.selectbox(
        "Tablo türü seçin:",
        [
            "Tek Makine (ProcessTime/DeliveryTime)",
            "Tek Makine + Öncelik (Lawler)",
            "İki Makine (Johnson: 2 Makine)",
            "Üç Makine (Johnson-3M: 3 Makine)",
            "Workstation (Received/Process/RemainingDue)",
        ]
    )

    # ✅ Mod değişimini takip et
    if "table_mode_prev" not in st.session_state:
        st.session_state.table_mode_prev = table_mode

    if st.session_state.table_mode_prev != table_mode:
        st.session_state.manual_df = None
        st.session_state.table_mode_prev = table_mode

    # ✅ Şablonlar
    if st.session_state.manual_df is None:
        if table_mode.startswith("Üç Makine"):
            st.session_state.manual_df = pd.DataFrame({
            "Job": [1, 2, 3, 4, 5, 6],
            "Machine1 Process Time": [4, 6, 3, 5, 8, 4],
            "Machine2 Process Time": [1, 2, 1, 3, 2, 1],
            "Machine3 Process Time": [3, 9, 2, 7, 6, 1],
    })
        elif table_mode.startswith("İki Makine"):
            st.session_state.manual_df = pd.DataFrame({
            "Job": ["A", "B", "C", "D", "E"],
            "Machine1 Process Time": [35, 15, 60, 50, 30],
            "Machine2 Process Time": [40, 20, 25, 45, 20],
    })

        elif table_mode.startswith("Workstation"):
            st.session_state.manual_df = pd.DataFrame({
            "Job": ["A","B","C","D","E","F"],
            "TimeSinceOrderReceived": [15,12,5,10,0,7],
            "ProcessTime": [25,16,14,10,12,16],
            "TimeRemainingUntilDeliveryDate": [29,27,68,48,80,47],
        })

        else:
    # Tek makine şablonları
            if table_mode.startswith("Tek Makine + Öncelik"):
        # Lawler örneği (hocanın slaytına benzer)
                st.session_state.manual_df = pd.DataFrame({
                "Job": ["J1", "J2", "J3", "J4", "J5", "J6"],
                "DeliveryTime": [3, 6, 9, 7, 11, 7],
                "ProcessTime": [2, 3, 4, 3, 2, 1],
            # Bu işten önce yapılması gereken işler (virgülle yazılabilir)
                "Predecessors": ["", "J1", "J2", "", "J4", "J4"],
        })
            else:
        # Normal tek makine (precedence yok)
                st.session_state.manual_df = pd.DataFrame({
                "Job": [1, 2, 3],
                "DeliveryTime": [10, 20, 15],
                "ProcessTime": [5, 7, 3],
        })

    # ✅ edited_manual burada tanımlanıyor
    edited_manual = st.data_editor(
        st.session_state.manual_df,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_manual"
    )

    # ✅ BUTON MUTLAKA BURADA OLMALI (edited_manual'dan sonra)
if st.button("✅ Manuel Tabloyu Kaydet ve Kullan"):

    # Johnson ve Workstation tabloları
    if table_mode.startswith(("İki Makine", "Üç Makine", "Workstation")):
        fixed = edited_manual.copy()

    # Lawler modu (precedence içeren)
    elif table_mode.startswith("Tek Makine + Öncelik"):
        fixed = edited_manual.copy()

        # sayısal kolonları temizle
        fixed["DeliveryTime"] = pd.to_numeric(fixed["DeliveryTime"], errors="coerce")
        fixed["ProcessTime"] = pd.to_numeric(fixed["ProcessTime"], errors="coerce")

        fixed = fixed.dropna(subset=["DeliveryTime", "ProcessTime"]).reset_index(drop=True)

    # Normal tek makine (SPT / EDD / LPT vs)
    else:
        fixed = finalize_df(edited_manual)

    # Tablo kontrolü
    if fixed is None or fixed.empty:
        st.error("⚠️ Tablo geçersiz. Lütfen gerekli sütunları doldurun.")

    else:
        st.session_state.manual_df = edited_manual.copy()
        st.session_state.df = fixed

        st.success("✅ Manuel tablo kaydedildi ve algoritmalara hazır.")
        st.dataframe(fixed, use_container_width=True)
# =========================================================
# 3) ALGORİTMALAR
# =========================================================
df = st.session_state.df

if df is not None and not df.empty:
    st.subheader("📄 Kullanılacak Tablo (Son Kaydedilen)")
    st.dataframe(df, use_container_width=True)

    st.markdown("### ⚙️ Sıralama Algoritması Seçin")
    algo_map = {
        "Moore-Hodgson Algoritması (Minimum Geciken İş)": "MOORE",
        "Shortest Processing Time (En Kısa İşlem Süresi) - SPT": "SPT",
        "Earliest Due Date (En Erken Teslim Tarihi) - EDD": "EDD",
        "Longest Processing Time (En Uzun İşlem Süresi) - LPT": "LPT",
        "First In First Out (İlk Gelen İlk Çıkar) - FIFO": "FIFO",
        "Last In First Out (Son Gelen İlk Çıkar) - LIFO": "LIFO",
        "Critical Ratio (Kritik Oran) - CR": "CR",
        "Modified Due Date (Modifiye Teslim Tarihi) - MDD": "MDD",
        "Johnson Algoritması (2 Makine | Makespan Min) - JOHNSON": "JOHNSON",
        "Johnson Algoritması (3 Makine | Koşullu) - JOHNSON3": "JOHNSON3",
        "Smith Algoritması (Tmax=0 | Ortalama Completion Min) - SMITH": "SMITH",
        "Workstation - FIFO (First come first served)": "WS_FIFO",
        "Workstation - EDD (Earliest delivery date first)": "WS_EDD",
        "Workstation - SPT (Shortest processing time first)": "WS_SPT",
        "Workstation - LPT (Longest processing time first)": "WS_LPT",
        "Workstation - LIFO (Last come first served)": "WS_LIFO",
        "Workstation - SRIT (Shortest remaining idle time first)": "WS_SRIT",
        "Lawler Algoritması (1|prec|Lmax | Max Gecikme Min) - LAWLER": "LAWLER",
    }

    label = st.selectbox("Algoritma", list(algo_map.keys()))
    algo = algo_map[label]

    save_result = st.checkbox("✅ Bu algoritma sonucunu karşılaştırma için kaydet", value=True)
if st.button("🚀 Optimum Tabloyu Hesapla"):
    try:
        optimal = None
        rejected = pd.DataFrame()

        if algo == "MOORE":
            optimal, rejected, final_df = moore_algorithm(df)
            st.success("✅ Moore sıralaması hesaplandı!")
            st.subheader("📌 Moore Final Sıra (Optimal + Rejected)")
            st.dataframe(final_df, use_container_width=True)
            if not rejected.empty:
                st.subheader("❌ Zamanında Yetişmeyen/Çıkarılan İşler (Rejected)")
                st.dataframe(rejected, use_container_width=True)
            if save_result:
                sumC = makespan(final_df)
                st.session_state.results[algo] = {"label": label, "df": final_df.copy(), "sumC": sumC}
                st.success(f"✅ Kaydedildi: {label} | Toplam Completion (∑Ci) = {sumC:.2f}")
            st.stop()

        elif algo == "JOHNSON":
            required_cols = {"Machine1 Process Time", "Machine2 Process Time"}
            if not required_cols.issubset(set(df.columns)):
                st.error(
                    "❌ Johnson (2 Makine) için tablo sütunları eksik.\n\n"
                    "Gerekli sütunlar:\n"
                    "- Machine1 Process Time\n"
                    "- Machine2 Process Time\n\n"
                    "Lütfen 'İki Makine (Johnson: 2 Makine)' tablosunu seçip kaydedin."
                )
                st.stop()

            res = johnson_algorithm(df)
            if res is None:
                st.error("❌ Johnson algoritması sonuç üretmedi (None döndü). Tablo formatını kontrol edin.")
                st.stop()

            seq_df, schedule_df, cmax, total_wait = res  # ✅ makespan ismini ezmiyoruz

            st.success("✅ Johnson sıralaması hesaplandı!")
            st.subheader("📌 Johnson Optimal Sıra")
            st.write(" - ".join(seq_df["Job"].astype(str).tolist()))

            st.subheader("📌 Johnson Çözüm Tablosu")
            st.dataframe(schedule_df, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Makespan (Cmax)", f"{cmax:.2f}")
            c2.metric("Toplam Atıl Süre (M2 Bekleme)", f"{total_wait:.2f}")
            if save_result:
                st.session_state.results[algo] = {"label": label, "df": schedule_df.copy(), "sumC": cmax}
                st.success(f"✅ Kaydedildi: {label} | Makespan (Cmax) = {cmax:.2f}")
            render_comparison_panel()
            st.stop()

        elif algo == "JOHNSON3":
            seq_df, schedule_df, makespan, idle_m2, idle_m3 = johnson_3machine_algorithm(df)

            st.success("✅ Johnson (3 Makine) sıralaması hesaplandı!")
            st.subheader("📌 Johnson-3M Optimal Sıra")
            st.write(" - ".join(seq_df["Job"].astype(str).tolist()))
            st.subheader("📌 Johnson-3M Çözüm Tablosu")
            st.dataframe(schedule_df, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Makespan (Cmax)", f"{makespan:.2f}")
            c2.metric("M2 Toplam Atıl Süre", f"{idle_m2:.2f}")
            c3.metric("M3 Toplam Atıl Süre", f"{idle_m3:.2f}")

            if save_result:
                st.session_state.results[algo] = {"label": label, "df": schedule_df.copy(), "sumC": makespan}
                st.success(f"✅ Kaydedildi: {label} | Makespan (Cmax) = {makespan:.2f}")
            st.stop()

        elif algo == "SPT":
            optimal, rejected = spt_algorithm(df)

        elif algo.startswith("WS_"):
            # 6 kural -> schedule + metrik
            if algo == "WS_FIFO":
                schedule_df, avg_delay, avg_lead, delayed_jobs = ws_fifo(df)
                rule_name = "FIFO"
            elif algo == "WS_EDD":
                schedule_df, avg_delay, avg_lead, delayed_jobs = ws_edd(df)
                rule_name = "EDD"
            elif algo == "WS_SPT":
                schedule_df, avg_delay, avg_lead, delayed_jobs = ws_spt(df)
                rule_name = "SPT"
            elif algo == "WS_LPT":
                schedule_df, avg_delay, avg_lead, delayed_jobs = ws_lpt(df)
                rule_name = "LPT"
            elif algo == "WS_LIFO":
                schedule_df, avg_delay, avg_lead, delayed_jobs = ws_lifo(df)
                rule_name = "LIFO"
            elif algo == "WS_SRIT":
                schedule_df, avg_delay, avg_lead, delayed_jobs = ws_srit(df)
                rule_name = "SRIT"
            else:
                raise ValueError("Bilinmeyen Workstation kuralı")

            st.success(f"✅ Workstation ({rule_name}) sıralaması hesaplandı!")
            st.subheader("📌 İş Sırası")
            order_col = "Order" if "Order" in schedule_df.columns else "Job"
            st.write(" - ".join(schedule_df[order_col].astype(str).tolist()))
            st.subheader("📌 Çözüm Tablosu")
            st.dataframe(schedule_df, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Average Delay (days)", f"{avg_delay:.2f}")
            c2.metric("Average Lead Time (days)", f"{avg_lead:.2f}")
            c3.metric("Number of Delayed Jobs", f"{delayed_jobs:d}")

            if save_result:
                # Karşılaştırma için 3 metriği sakla
                st.session_state.results[algo] = {
                    "label": label,
                    "df": schedule_df.copy(),
                    "avg_delay": round(avg_delay, 2),
                    "avg_lead": round(avg_lead, 2),
                    "delayed_jobs": delayed_jobs,
                    "sumC": avg_lead
                }
                st.success(f"✅ Kaydedildi: {label}")
            render_comparison_panel()
            st.stop()

        elif algo == "LAWLER":
            optimal, rejected = lawler_algorithm(df)  # ✅ önce algoritmayı çalıştır

            if optimal is None or optimal.empty:
                raise ValueError("Lawler çıktı üretmedi. Tabloyu ve Predecessors girişini kontrol et.")

            # ✅ Lmax hesapla (algoritmayı bozmuyor, sadece çıktı metriği)
            p = pd.to_numeric(optimal["ProcessTime"], errors="coerce").fillna(0).astype(float).values
            d = pd.to_numeric(optimal["DeliveryTime"], errors="coerce").fillna(0).astype(float).values
            C = p.cumsum()
            L = C - d
            Lmax = float(np.max(L)) if len(L) > 0 else float("inf")

            st.success("✅ Lawler sıralaması hesaplandı!")
            st.subheader("📌 Lawler Optimal Sıra")
            st.write(" - ".join(optimal["Job"].astype(str).tolist()))
            st.subheader("📌 Optimum Sıralama Tablosu")
            st.dataframe(optimal, use_container_width=True)
            

            # ✅ Kaydet: panel sumC bekliyor diye buraya Lmax yazıyoruz
            if save_result:
                st.session_state.results[algo] = {
                    "label": label,
                    "df": optimal.copy(),
                    "sumC": Lmax
                }
                

            st.stop()

        elif algo == "EDD":
            optimal, rejected = edd_algorithm(df)

        elif algo == "LPT":
            optimal, rejected = lpt_algorithm(df)

        elif algo == "FIFO":
            optimal, rejected = fcfs_algorithm(df)

        elif algo == "LIFO":
            optimal, rejected = lifo_algorithm(df)

        elif algo == "CR":
            optimal, rejected = cr_algorithm(df)

        elif algo == "MDD":
            optimal, rejected = mdd_algorithm(df)

        elif algo == "SMITH":
            optimal, rejected = smith_algorithm(df)

        else:
            raise ValueError(f"Bilinmeyen algoritma kodu: {algo}")

        st.success("✅ Optimum sıralama hesaplandı!")
        st.subheader("📌 Optimum Sıralama Tablosu")
        st.dataframe(optimal, use_container_width=True)

        if rejected is not None and not rejected.empty:
            st.subheader("❌ Rejected")
            st.dataframe(rejected, use_container_width=True)
        else:
            st.info("⏱ Bu yöntemde 'rejected' üretilmez (boş döner).")

        if save_result:
            sumC = total_completion_time(optimal)
            st.session_state.results[algo] = {
                "label": label,
                "df": optimal.copy(),
                "sumC": sumC
            }
            st.success(f"✅ Kaydedildi: {label} | Toplam Completion (∑Ci) = {sumC:.2f}")

    except Exception as e:
        st.error(f"❌ Hata: {e}")

render_comparison_panel()
st.subheader("Bu uygulama, bitirme projesi kapsamında geliştirilmiştir. Destekleri için değerli hocamız Dr. Öğretim Üyesi Üzeyir Pala'ya teşekkür ederiz.")

