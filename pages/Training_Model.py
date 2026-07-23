import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_utils import (
    FEATURES,
    TARGET,
    DEFAULT_TIME_STEPS,
    scale_features,
    create_sequences,
    train_test_split_sequences,
)
from utils.model_utils import (
    train_model,
    evaluate_model,
    save_artifacts,
    load_artifacts,
    artifacts_exist,
    StreamlitProgressCallback,
)

st.set_page_config(page_title="Training Model", page_icon="🧠", layout="wide")
st.title("Training & Evaluasi Model GRU")

if st.session_state.get("df_clean") is None:
    st.warning("Data belum dimuat. Silakan kembali ke halaman **Beranda** dan muat data terlebih dahulu.")
    st.stop()

df = st.session_state["df_clean"]

# ==========================================================
# TAHAPAN PIPELINE (dokumentasi visual)
# ==========================================================
st.subheader("Tahapan Training")
steps = st.columns(5)
step_labels = [
    ("1️⃣ Cleaning", "Interpolasi harga harian, forward-fill data bulanan, buang outlier USDIDR < 5000"),
    ("2️⃣ Feature Engineering", "Lag 1/7/30 hari untuk USDIDR, lag OIL/GOLD/SP500/IHSG/VIX"),
    ("3️⃣ Scaling & Windowing", "MinMaxScaler + sliding window 30 hari"),
    ("4️⃣ Train/Test Split", "80% train, 20% test (berurutan berdasarkan waktu)"),
    ("5️⃣ Training GRU", "GRU(64)→Dropout→GRU(32)→Dropout→Dense(16)→Dense(1)"),
]
for col, (title, desc) in zip(steps, step_labels):
    with col:
        st.markdown(f"**{title}**")
        st.caption(desc)

with st.expander("Lihat kode arsitektur model"):
    st.code(
        '''model = Sequential([
    GRU(64, return_sequences=True, input_shape=(TIME_STEPS, n_features)),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=tf.keras.losses.Huber(),
    metrics=["mae", "mse"]
)''',
        language="python",
    )

st.markdown("---")

# ==========================================================
# KONFIGURASI TRAINING
# ==========================================================
st.subheader("Konfigurasi Training")

c1, c2, c3, c4 = st.columns(4)
time_steps = c1.number_input("Time steps (hari)", min_value=7, max_value=60, value=DEFAULT_TIME_STEPS, step=1)
epochs = c2.number_input("Epochs maksimum", min_value=5, max_value=300, value=50, step=5)
batch_size = c3.selectbox("Batch size", [16, 32, 64, 128], index=1)
train_ratio = c4.slider("Proporsi data train", 0.5, 0.9, 0.8, 0.05)

use_saved = False
if artifacts_exist():
    use_saved = st.checkbox(
        "Gunakan model tersimpan (skip training) jika tersedia", value=True
    )

col_a, col_b = st.columns([1, 1])
run_training = col_a.button("Mulai Training", type="primary", use_container_width=True)
load_existing = col_b.button("Muat Model Tersimpan", use_container_width=True, disabled=not artifacts_exist())

# ==========================================================
# EKSEKUSI TRAINING
# ==========================================================
if run_training:
    with st.spinner("Menyiapkan data (scaling & windowing)..."):
        X, y, feature_scaler, target_scaler = scale_features(df, FEATURES, TARGET)
        X_seq, y_seq = create_sequences(X, y, time_steps)
        X_train, X_test, y_train, y_test, train_size = train_test_split_sequences(X_seq, y_seq, train_ratio)

    st.info(f"Shape data — Train: {X_train.shape} · Test: {X_test.shape}")

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    chart_placeholder = st.empty()

    callback = StreamlitProgressCallback(progress_bar, status_text, chart_placeholder, epochs)

    with st.spinner("Training sedang berjalan, mohon tunggu..."):
        model, history = train_model(
            X_train,
            y_train,
            input_shape=(X_train.shape[1], X_train.shape[2]),
            epochs=epochs,
            batch_size=batch_size,
            progress_callback=callback,
        )

    st.success("✅ Training selesai!")

    save_artifacts(model, feature_scaler, target_scaler, FEATURES, time_steps)
    load_artifacts.clear()  # invalidate cache resource so reload picks up new model

    st.session_state["model_bundle"] = (model, feature_scaler, target_scaler, FEATURES, time_steps)
    st.session_state["eval_data"] = evaluate_model(model, X_test, y_test, target_scaler)
    st.session_state["train_size"] = train_size
    st.session_state["dates_test"] = df["Date"].iloc[time_steps + train_size :].reset_index(drop=True)

elif load_existing or (use_saved and "model_bundle" not in st.session_state and artifacts_exist()):
    with st.spinner("Memuat model tersimpan..."):
        model, feature_scaler, target_scaler, features, ts = load_artifacts()
        # gunakan scaler yang sudah tersimpan (fit saat training), bukan fit baru
        X = feature_scaler.transform(df[features])
        y = target_scaler.transform(df[[TARGET]])
        X_seq, y_seq = create_sequences(X, y, ts)
        X_train, X_test, y_train, y_test, train_size = train_test_split_sequences(X_seq, y_seq, train_ratio)

    st.session_state["model_bundle"] = (model, feature_scaler, target_scaler, features, ts)
    st.session_state["eval_data"] = evaluate_model(model, X_test, y_test, target_scaler)
    st.session_state["train_size"] = train_size
    st.session_state["dates_test"] = df["Date"].iloc[ts + train_size :].reset_index(drop=True)
    st.success("✅ Model tersimpan berhasil dimuat.")

# ==========================================================
# HASIL EVALUASI
# ==========================================================
if st.session_state.get("eval_data") is not None:
    st.markdown("---")
    st.subheader("📈 Hasil Evaluasi pada Data Test")

    ev = st.session_state["eval_data"]
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{ev['mae']:.2f}")
    m2.metric("RMSE", f"{ev['rmse']:.2f}")
    m3.metric("R²", f"{ev['r2']:.4f}")

    dates_test = st.session_state.get("dates_test")
    y_true = ev["y_true"].flatten()
    y_pred = ev["y_pred"].flatten()

    if dates_test is not None and len(dates_test) == len(y_true):
        x_axis = dates_test
    else:
        x_axis = np.arange(len(y_true))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=y_true, name="Aktual", mode="lines"))
    fig.add_trace(go.Scatter(x=x_axis, y=y_pred, name="Prediksi", mode="lines"))
    fig.update_layout(title="Aktual vs Prediksi USDIDR (Data Test)", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Prediksi", opacity=0.5))
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    fig_scatter.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Ideal (y=x)", line=dict(dash="dash")))
    fig_scatter.update_layout(title="Scatter Plot Aktual vs Prediksi", xaxis_title="Aktual", yaxis_title="Prediksi", height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Klik **Mulai Training** untuk melatih model baru, atau **Muat Model Tersimpan** jika sudah ada model sebelumnya.")
