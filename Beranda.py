import streamlit as st
import pandas as pd

from utils.data_utils import load_raw_data, prepare_dataset

st.set_page_config(
    page_title="USDIDR GRU Forecast",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# SIDEBAR — SUMBER DATA (dipakai di semua halaman via session_state)
# ==========================================================
st.sidebar.title("USDIDR GRU Forecast")
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Sumber Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (opsional)",
    type=["csv"],
    help=(
        "Kolom minimal: Date, USDIDR, OIL, GOLD, SP500, IHSG, VIX, CPI, "
        "BI_rate, US_rate. Jika tidak diupload, aplikasi akan mencoba "
        "mengunduh dataset asli dari Kaggle (butuh kredensial)."
    ),
)

if st.sidebar.button("Muat / Refresh Data", use_container_width=True) or "df_raw" not in st.session_state:
    with st.spinner("Memuat data..."):
        df_raw, source = load_raw_data(uploaded_file)
    if df_raw is None:
        st.session_state["df_raw"] = None
        st.session_state["data_error"] = source
    else:
        st.session_state["df_raw"] = df_raw
        st.session_state["data_source"] = source
        st.session_state["data_error"] = None
        try:
            st.session_state["df_clean"] = prepare_dataset(df_raw)
        except Exception as e:
            st.session_state["df_clean"] = None
            st.session_state["data_error"] = f"Gagal melakukan preprocessing: {e}"

st.sidebar.markdown("---")
st.sidebar.info(
    "Gunakan menu di atas sidebar ini untuk berpindah halaman:\n\n"
    "- **Analisis Data** — grafik historis GOLD, OIL, dll\n"
    "- **Training Model** — tahapan & proses pelatihan GRU\n"
    "- **Prediksi & Forecast** — forecast USDIDR N hari ke depan\n"
    "- **Simulasi Interaktif** — what-if scenario real-time"
)

# ==========================================================
# HALAMAN UTAMA
# ==========================================================

st.title("Dashboard Forecast USDIDR Metode GRU")
st.markdown(
    """
Selamat datang di dashboard interaktif **forecasting nilai tukar USD/IDR**
menggunakan model **GRU (Gated Recurrent Unit)**. Dashboard ini mereplikasi
pipeline notebook training asli — mulai dari pembersihan data, feature
engineering (lag & moving average), hingga pelatihan dan evaluasi model —
dan membungkusnya menjadi aplikasi yang bisa dieksplorasi secara interaktif.
"""
)

if st.session_state.get("data_error"):
    st.warning(st.session_state["data_error"])
elif st.session_state.get("df_clean") is not None:
    df = st.session_state["df_clean"]
    source_label = {"upload": "File upload", "kagglehub": "Kaggle (raphaelnazareth/indonesia-financial-time-series-dataset-2010-2026)"}
    st.success(
        f"Data berhasil dimuat dari **{source_label.get(st.session_state.get('data_source'), 'sumber tidak diketahui')}** "
        f"— {len(df):,} baris siap pakai (setelah cleaning & feature engineering)."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rentang Data", f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    col2.metric("USDIDR Terakhir", f"Rp {df['USDIDR'].iloc[-1]:,.0f}")
    col3.metric("GOLD Terakhir", f"${df['GOLD'].iloc[-1]:,.2f}")
    col4.metric("OIL Terakhir", f"${df['OIL'].iloc[-1]:,.2f}")

    with st.expander("Lihat cuplikan data (5 baris terakhir)"):
        st.dataframe(df.tail(), use_container_width=True)

 # st.markdown("### 🧭 Arsitektur Pipeline")
# st.markdown(
#     """
#     ```
#     Data Mentah (Date, USDIDR, OIL, GOLD, SP500, IHSG, VIX, CPI, BI_rate, US_rate)
#                 │
#                 ▼
#     Cleaning: interpolasi harian, forward-fill bulanan, filter outlier
#                 │
#                 ▼
#     Feature Engineering: lag 1/7/30 hari, moving average, rolling std
#                 │
#                 ▼
#     Scaling (MinMaxScaler) + Windowing (30 hari) → Sequence
#                 │
#                 ▼
#     GRU(64) → Dropout → GRU(32) → Dropout → Dense(16) → Dense(1)
#                 │
#                 ▼
#     Prediksi USDIDR hari berikutnya
#     ```
#     """
# )
else:
    st.info("⬅ Silakan upload file CSV pada sidebar, atau klik **Muat / Refresh Data** untuk mencoba memuat dataset default.")

st.markdown("---")
st.caption(
    "Model: GRU 2-layer · Fitur: 18 variabel (harga + lag) · Target: USDIDR · "
    "Dibangun dengan Streamlit & TensorFlow/Keras."
)
