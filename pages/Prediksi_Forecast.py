import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.model_utils import artifacts_exist, load_artifacts, recursive_forecast

st.set_page_config(page_title="Prediksi & Forecast", page_icon="🔮", layout="wide")
st.title("Prediksi & Forecast USDIDR")

if st.session_state.get("df_clean") is None:
    st.warning("Data belum dimuat. Silakan kembali ke halaman **Beranda** dan muat data terlebih dahulu.")
    st.stop()

df = st.session_state["df_clean"]

# ==========================================================
# AMBIL MODEL (dari session_state hasil training, atau dari file tersimpan)
# ==========================================================
model_bundle = st.session_state.get("model_bundle")
if model_bundle is None and artifacts_exist():
    model_bundle = load_artifacts()
    st.session_state["model_bundle"] = model_bundle

if model_bundle is None:
    st.warning(
        "Model belum tersedia. Silakan latih model terlebih dahulu di halaman "
        "**Training Model**, atau pastikan file model tersimpan ada di folder `model/`."
    )
    st.stop()

model, feature_scaler, target_scaler, features, time_steps = model_bundle

last_row = df.iloc[-1]

st.markdown(
    f"Data historis terakhir: **{last_row['Date'].date()}** — "
    f"USDIDR: **Rp {last_row['USDIDR']:,.0f}**, GOLD: **${last_row['GOLD']:,.2f}**, "
    f"OIL: **${last_row['OIL']:,.2f}**"
)

st.markdown("---")

# ==========================================================
# INPUT JUMLAH HARI
# ==========================================================
st.subheader("1. Tentukan Horizon Forecast")
n_days = st.slider("Jumlah hari ke depan yang ingin diprediksi", min_value=1, max_value=60, value=14)

st.markdown("---")

# ==========================================================
# INPUT ASUMSI GOLD & OIL (per hari, bisa diedit manual)
# ==========================================================
st.subheader("2. Asumsi Harga GOLD & OIL untuk Setiap Hari ke Depan")

mode = st.radio(
    "Metode input asumsi",
    ["Nilai tetap (flat)", "Tren % per hari", "Edit manual per hari"],
    horizontal=True,
)

future_dates = [last_row["Date"] + pd.Timedelta(days=i + 1) for i in range(n_days)]

if mode == "Nilai tetap (flat)":
    c1, c2 = st.columns(2)
    gold_flat = c1.number_input("Harga GOLD tetap (USD)", value=float(last_row["GOLD"]), step=1.0)
    oil_flat = c2.number_input("Harga OIL tetap (USD)", value=float(last_row["OIL"]), step=0.5)
    future_gold = [gold_flat] * n_days
    future_oil = [oil_flat] * n_days

elif mode == "Tren % per hari":
    c1, c2 = st.columns(2)
    gold_trend = c1.number_input("Perubahan GOLD (%/hari)", value=0.0, step=0.05, format="%.2f")
    oil_trend = c2.number_input("Perubahan OIL (%/hari)", value=0.0, step=0.05, format="%.2f")
    future_gold = [float(last_row["GOLD"]) * (1 + gold_trend / 100) ** (i + 1) for i in range(n_days)]
    future_oil = [float(last_row["OIL"]) * (1 + oil_trend / 100) ** (i + 1) for i in range(n_days)]

else:
    default_df = pd.DataFrame(
        {
            "Tanggal": [d.date() for d in future_dates],
            "GOLD": [float(last_row["GOLD"])] * n_days,
            "OIL": [float(last_row["OIL"])] * n_days,
        }
    )
    edited = st.data_editor(
        default_df,
        use_container_width=True,
        disabled=["Tanggal"],
        hide_index=True,
        key="manual_editor",
    )
    future_gold = edited["GOLD"].tolist()
    future_oil = edited["OIL"].tolist()

with st.expander("Asumsi variabel makro lainnya (opsional)"):
    st.caption("Nilai lain (SP500, IHSG, VIX, CPI, BI rate, US rate) diasumsikan tetap kecuali diatur trennya di sini.")
    cc1, cc2, cc3 = st.columns(3)
    sp500_trend = cc1.number_input("SP500 (%/hari)", value=0.0, step=0.05, format="%.2f")
    ihsg_trend = cc2.number_input("IHSG (%/hari)", value=0.0, step=0.05, format="%.2f")
    vix_trend = cc3.number_input("VIX (%/hari)", value=0.0, step=0.05, format="%.2f")
    cc4, cc5, cc6 = st.columns(3)
    cpi_trend = cc4.number_input("CPI (%/hari)", value=0.0, step=0.01, format="%.3f")
    bi_trend = cc5.number_input("BI rate (%/hari)", value=0.0, step=0.01, format="%.3f")
    us_trend = cc6.number_input("US rate (%/hari)", value=0.0, step=0.01, format="%.3f")

exog_trends = {
    "SP500": sp500_trend,
    "IHSG": ihsg_trend,
    "VIX": vix_trend,
    "CPI": cpi_trend,
    "BI_rate": bi_trend,
    "US_rate": us_trend,
}

st.markdown("---")

# ==========================================================
# JALANKAN FORECAST
# ==========================================================
if st.button("🚀 Jalankan Forecast", type="primary"):
    with st.spinner(f"Menghitung forecast {n_days} hari ke depan secara rekursif..."):
        forecast_df = recursive_forecast(
            df,
            model,
            feature_scaler,
            target_scaler,
            features,
            time_steps,
            n_days,
            future_oil=future_oil,
            future_gold=future_gold,
            exog_trends=exog_trends,
        )
    st.session_state["forecast_df"] = forecast_df
    st.session_state["forecast_inputs"] = {"gold": future_gold, "oil": future_oil}

if st.session_state.get("forecast_df") is not None:
    forecast_df = st.session_state["forecast_df"]

    st.subheader("📈 3. Hasil Forecast")

    hist_tail = df[["Date", "USDIDR"]].tail(60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_tail["Date"], y=hist_tail["USDIDR"], name="Historis", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["USDIDR_Forecast"],
            name="Forecast",
            mode="lines+markers",
            line=dict(dash="dash", color="orangered"),
        )
    )
    fig.update_layout(title="Forecast USDIDR", height=480, hovermode="x unified", yaxis_title="Rp")
    st.plotly_chart(fig, use_container_width=True)

    last_actual = df["USDIDR"].iloc[-1]
    last_forecast = forecast_df["USDIDR_Forecast"].iloc[-1]
    delta = last_forecast - last_actual
    delta_pct = delta / last_actual * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("USDIDR Saat Ini", f"Rp {last_actual:,.0f}")
    m2.metric(
        f"Forecast Rp (hari ke-{n_days})",
        f"Rp {last_forecast:,.0f}",
        delta=f"{delta:+,.0f} ({delta_pct:+.2f}%)",
    )
    m3.metric("Rata-rata Forecast", f"Rp {forecast_df['USDIDR_Forecast'].mean():,.0f}")

    with st.expander("📋 Tabel detail forecast"):
        display_df = forecast_df.copy()
        display_df["Date"] = display_df["Date"].dt.date
        display_df["USDIDR_Forecast"] = display_df["USDIDR_Forecast"].round(0)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh hasil forecast (CSV)", csv, "forecast_usdidr.csv", "text/csv")

    st.caption(
        "⚠️ Forecast bersifat rekursif: hasil prediksi hari ke-N digunakan sebagai bagian "
        "input untuk memprediksi hari ke-N+1. Semakin panjang horizon, semakin besar "
        "potensi akumulasi error. Nilai GOLD/OIL/dll adalah **asumsi skenario**, bukan data aktual."
    )
else:
    st.info("Atur horizon & asumsi di atas, lalu klik **Jalankan Forecast**.")
