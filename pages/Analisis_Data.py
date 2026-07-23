import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Analisis Data", page_icon="📊", layout="wide")

st.title("Analisis Data Historis")

if st.session_state.get("df_clean") is None:
    st.warning("Data belum dimuat. Silakan kembali ke halaman **Beranda** dan muat data terlebih dahulu.")
    st.stop()

df = st.session_state["df_clean"]

# ==========================================================
# FILTER TANGGAL
# ==========================================================
st.sidebar.subheader("Filter Rentang Tanggal")
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Pilih rentang", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
dff = df.loc[mask]

variables = {
    "USDIDR": "USD/IDR (Rp)",
    "GOLD": "Harga Emas (USD)",
    "OIL": "Harga Minyak (USD)",
    "SP500": "Indeks S&P 500",
    "IHSG": "Indeks IHSG",
    "VIX": "Indeks Volatilitas (VIX)",
    "CPI": "CPI (Inflasi)",
    "BI_rate": "Suku Bunga BI (%)",
    "US_rate": "Suku Bunga The Fed (%)",
}

st.sidebar.subheader("Pilih Variabel")
selected_vars = st.sidebar.multiselect(
    "Variabel yang ditampilkan", list(variables.keys()), default=["USDIDR", "GOLD", "OIL"]
)

# ==========================================================
# GRAFIK UTAMA
# ==========================================================
st.subheader("Tren Historis")
tabs = st.tabs([variables[v] for v in selected_vars] if selected_vars else ["Pilih variabel"])

for tab, var in zip(tabs, selected_vars):
    with tab:
        fig = px.line(dff, x="Date", y=var, title=variables[var])
        fig.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Terakhir", f"{dff[var].iloc[-1]:,.2f}")
        c2.metric("Rata-rata", f"{dff[var].mean():,.2f}")
        c3.metric("Min", f"{dff[var].min():,.2f}")
        c4.metric("Max", f"{dff[var].max():,.2f}")

# ==========================================================
# PERBANDINGAN MULTI-VARIABEL (NORMALIZED)
# ==========================================================
st.markdown("---")
st.subheader("Perbandingan Tren (Normalisasi 0-1)")
if len(selected_vars) >= 2:
    norm_df = dff[["Date"] + selected_vars].copy()
    for v in selected_vars:
        rng = norm_df[v].max() - norm_df[v].min()
        norm_df[v] = (norm_df[v] - norm_df[v].min()) / rng if rng != 0 else 0

    fig2 = go.Figure()
    for v in selected_vars:
        fig2.add_trace(go.Scatter(x=norm_df["Date"], y=norm_df[v], name=variables[v], mode="lines"))
    fig2.update_layout(height=450, hovermode="x unified", yaxis_title="Skala Normalisasi")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Pilih minimal 2 variabel di sidebar untuk melihat perbandingan tren.")

# ==========================================================
# KORELASI
# ==========================================================
st.markdown("---")
st.subheader("Matriks Korelasi Antar Variabel")
corr_cols = list(variables.keys())
corr = dff[corr_cols].corr()
fig3 = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
)
fig3.update_layout(height=500)
st.plotly_chart(fig3, use_container_width=True)

st.caption(
    "Korelasi dihitung dari data pada rentang tanggal terpilih. Nilai mendekati "
    "1 atau -1 menunjukkan hubungan linear yang kuat."
)

# ==========================================================
# STATISTIK DESKRIPTIF
# ==========================================================
st.markdown("---")
st.subheader("Statistik Deskriptif")
st.dataframe(dff[corr_cols].describe().T, use_container_width=True)
