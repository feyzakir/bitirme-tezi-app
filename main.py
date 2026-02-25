import streamlit as st
import pandas as pd

from table import extract_jobs_from_image, finalize_df
from algorithm import (
    moore_algorithm, spt_algorithm, edd_algorithm, lpt_algorithm, fcfs_algorithm,lifo_algorithm, cr_algorithm, mdd_algorithm, johnson_algorithm, smith_algorithm 
)

def total_completion_time(single_machine_df: pd.DataFrame) -> float:
    if single_machine_df is None or single_machine_df.empty:
        return float("inf")
    p = pd.to_numeric(single_machine_df["ProcessTime"], errors="coerce").fillna(0).astype(float).values
    C = p.cumsum()
    return float(C.sum())


st.set_page_config(page_title="İş Sıralama Optimizasyonu", layout="centered")
st.title("İş Sıralama ve Üretim Planlama Aracı")

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
        ["Tek Makine (ProcessTime/DeliveryTime)",
         "İki Makine (Johnson: M1Time/M2Time)"]
    )

    # ✅ Mod değişimini takip et (ilk kez yoksa oluştur)
    if "table_mode_prev" not in st.session_state:
        st.session_state.table_mode_prev = table_mode

    # ✅ Mod değiştiyse tabloyu şablona çevir
    if st.session_state.table_mode_prev != table_mode:
        st.session_state.manual_df = None
        st.session_state.table_mode_prev = table_mode

    # İlk açılış / mod değişimi sonrası şablon
    if st.session_state.manual_df is None:
        if table_mode.startswith("İki Makine"):
            st.session_state.manual_df = pd.DataFrame({
                "Job": ["A", "B", "C", "D", "E"],
                "M1Time": [35, 15, 60, 50, 30],
                "M2Time": [40, 20, 25, 45, 20],
            })
        else:
            st.session_state.manual_df = pd.DataFrame({
                "Job": [1, 2, 3],
                "DeliveryTime": [10, 20, 15],
                "ProcessTime": [5, 7, 3],
            })

    edited_manual = st.data_editor(
        st.session_state.manual_df,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_manual"
    )

    if st.button("✅ Manuel Tabloyu Kaydet ve Kullan"):
        # ⚠️ finalize_df büyük ihtimalle DeliveryTime/ProcessTime istiyor.
        # Johnson modunda finalize_df boş dönebilir.
        fixed = finalize_df(edited_manual)

        if fixed.empty:
            st.error("⚠️ Tablo geçersiz. (Tek makinede DeliveryTime/ProcessTime, Johnson’da M1Time/M2Time dolu olmalı.)")
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
        "Smith Algoritması (Tmax=0 | Ortalama Completion Min) - SMITH": "SMITH",
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
                    sumC = total_completion_time(final_df)
                    st.session_state.results[algo] = {"label": label, "df": final_df.copy(), "sumC": sumC}
                    st.success(f"✅ Kaydedildi: {label} | Toplam Completion (∑Ci) = {sumC:.2f}")
                st.stop()
            elif algo == "JOHNSON":
                seq_df, schedule_df, makespan, total_wait = johnson_algorithm(df)

                st.success("✅ Johnson sıralaması hesaplandı!")
                st.subheader("📌 Johnson Optimal Sıra")
                st.write(" - ".join(seq_df["Job"].astype(str).tolist()))

                st.subheader("📌 Johnson Çözüm Tablosu")
                st.dataframe(schedule_df, use_container_width=True)

                c1, c2 = st.columns(2)
                c1.metric("Makespan (Cmax)", f"{makespan:.2f}")
                c2.metric("Toplam Atıl Süre (M2 Bekleme)", f"{total_wait:.2f}")

                if save_result:
        # Karşılaştırmada Johnson için Cmax saklıyoruz
                    st.session_state.results[algo] = {"label": label, "df": schedule_df.copy(), "sumC": makespan}
                    st.success(f"✅ Kaydedildi: {label} | Makespan (Cmax) = {makespan:.2f}")

                st.stop()
            

            elif algo == "SPT":
                optimal, rejected = spt_algorithm(df)
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

    # =========================================================
    #  Karşılaştırma Paneli (Kaydedilen Sonuçlar)
    # =========================================================
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
            st.dataframe(summary, use_container_width=True)

            best_code = summary.loc[0, "Kod"]
            best = st.session_state.results[best_code]

            st.success(f"🏆 En iyi (en düşük ∑Ci): {best['label']}")
            st.subheader("✅ En iyi algoritmanın iş sırası sonucu")
            st.dataframe(best["df"], use_container_width=True)

else:
    st.info("Tablo yükleyin veya manuel tabloyu kaydedin. Sonra algoritma seçebilirsiniz.")

st.subheader("Bu uygulama, bitirme projesi kapsamında geliştirilmiştir. Destekleri için değerli hocamız Dr. Öğretim Üyesi Üzeyir Pala'ya teşekkür ederiz.")

