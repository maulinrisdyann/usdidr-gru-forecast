import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.model_utils import artifacts_exist, load_artifacts

st.set_page_config(page_title="Simulasi Interaktif", page_icon="🎮", layout="wide")
st.title("Simulasi Interaktif: What-If Scenario")

st.markdown(
    "Geser slider di bawah untuk mengubah kondisi pasar **hari ini** secara "
    "hipotetis, lalu lihat bagaimana prediksi USDIDR untuk **besok** bereaksi "
    "secara instan — cocok untuk eksplorasi sensitivitas model."
)

if st.session_state.get("df_clean") is None:
    st.warning("Data belum dimuat. Silakan kembali ke halaman **Beranda** dan muat data terlebih dahulu.")
    st.stop()

df = st.session_state["df_clean"]

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

st.markdown("---")
st.subheader("Ubah Kondisi Pasar (Persentase dari Nilai Terakhir)")

col1, col2, col3 = st.columns(3)
gold_pct = col1.slider("GOLD (%)", -20, 20, 0, help=f"Nilai terakhir: ${last_row['GOLD']:,.2f}")
oil_pct = col2.slider("OIL (%)", -30, 30, 0, help=f"Nilai terakhir: ${last_row['OIL']:,.2f}")
sp500_pct = col3.slider("SP500 (%)", -15, 15, 0, help=f"Nilai terakhir: {last_row['SP500']:,.2f}")

col4, col5, col6 = st.columns(3)
ihsg_pct = col4.slider("IHSG (%)", -15, 15, 0, help=f"Nilai terakhir: {last_row['IHSG']:,.2f}")
vix_pct = col5.slider("VIX / Volatilitas (%)", -50, 100, 0, help=f"Nilai terakhir: {last_row['VIX']:,.2f}")
bi_pct = col6.slider("Suku Bunga BI (%)", -20, 20, 0, help=f"Nilai terakhir: {last_row['BI_rate']:,.2f}%")

# ==========================================================
# BANGUN WINDOW SIMULASI (ubah hanya baris terakhir sesuai slider)
# ==========================================================
window_df = df[features].tail(time_steps).copy()

sim_row = window_df.iloc[-1].copy()
sim_row["GOLD"] = last_row["GOLD"] * (1 + gold_pct / 100)
sim_row["OIL"] = last_row["OIL"] * (1 + oil_pct / 100)
sim_row["SP500"] = last_row["SP500"] * (1 + sp500_pct / 100)
sim_row["IHSG"] = last_row["IHSG"] * (1 + ihsg_pct / 100)
sim_row["VIX"] = last_row["VIX"] * (1 + vix_pct / 100)
sim_row["BI_rate"] = last_row["BI_rate"] * (1 + bi_pct / 100)
# selaraskan fitur lag GOLD/OIL/SP500/IHSG/VIX dengan nilai baru
sim_row["GOLD_Lag1"] = last_row["GOLD"]
sim_row["OIL_Lag1"] = last_row["OIL"]
sim_row["SP500_Lag1"] = last_row["SP500"]
sim_row["IHSG_Lag1"] = last_row["IHSG"]
sim_row["VIX_Lag1"] = last_row["VIX"]

window_sim = window_df.copy()
window_sim.iloc[-1] = sim_row

X_base = feature_scaler.transform(window_df).reshape(1, time_steps, len(features))
X_sim = feature_scaler.transform(window_sim).reshape(1, time_steps, len(features))

pred_base = target_scaler.inverse_transform(model.predict(X_base, verbose=0))[0, 0]
pred_sim = target_scaler.inverse_transform(model.predict(X_sim, verbose=0))[0, 0]

st.markdown("---")
st.subheader("Hasil Simulasi")

c1, c2, c3 = st.columns(3)
c1.metric("Prediksi Baseline (tanpa perubahan)", f"Rp {pred_base:,.0f}")
c2.metric(
    "Prediksi Skenario (dengan perubahan)",
    f"Rp {pred_sim:,.0f}",
    delta=f"{pred_sim - pred_base:+,.0f}",
)
direction = "🔺 Rupiah melemah" if pred_sim > pred_base else ("🔻 Rupiah menguat" if pred_sim < pred_base else "➡️ Tidak berubah")
c3.metric("Arah Pergerakan", direction)

# Gauge chart
fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number+delta",
        value=pred_sim,
        delta={"reference": pred_base},
        gauge={
            "axis": {"range": [pred_base * 0.9, pred_base * 1.1]},
            "bar": {"color": "orangered" if pred_sim > pred_base else "seagreen"},
            "steps": [
                {"range": [pred_base * 0.9, pred_base], "color": "#d4f7dc"},
                {"range": [pred_base, pred_base * 1.1], "color": "#fddede"},
            ],
        },
        title={"text": "Prediksi USDIDR Skenario"},
    )
)
fig_gauge.update_layout(height=350)
st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")
st.subheader("💱 Konverter Cepat")
amount_usd = st.number_input("Jumlah USD", min_value=0.0, value=100.0, step=10.0)
cc1, cc2 = st.columns(2)
cc1.metric("Konversi (Baseline)", f"Rp {amount_usd * pred_base:,.0f}")
cc2.metric("Konversi (Skenario)", f"Rp {amount_usd * pred_sim:,.0f}")

st.caption(
    "Simulasi ini mengubah nilai fitur pada hari terakhir dari window historis lalu "
    "meminta model memprediksi USDIDR hari berikutnya — berguna untuk memahami arah "
    "sensitivitas model terhadap tiap variabel, bukan sebagai forecast presisi."
)
