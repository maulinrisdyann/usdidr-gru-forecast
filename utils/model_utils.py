"""
model_utils.py
===============
Membangun, melatih, mengevaluasi, menyimpan/memuat, dan melakukan
forecast rekursif multi-hari untuk model GRU USDIDR.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

from .data_utils import FEATURES, TARGET, DEFAULT_TIME_STEPS

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "gru_usdidr.keras")
FEATURE_SCALER_PATH = os.path.join(ARTIFACT_DIR, "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(ARTIFACT_DIR, "target_scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "features.pkl")
TIMESTEPS_PATH = os.path.join(ARTIFACT_DIR, "timesteps.pkl")


# ==========================================================
# BUILD MODEL (arsitektur identik dengan notebook)
# ==========================================================

def build_gru_model(input_shape, gru_units=(64, 32), dense_units=16, dropout=0.2, lr=0.001):
    model = Sequential(
        [
            GRU(gru_units[0], return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            GRU(gru_units[1]),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(),
        metrics=["mae", "mse"],
    )
    return model


class StreamlitProgressCallback(Callback):
    """Callback Keras yang mendorong update live ke widget Streamlit."""

    def __init__(self, progress_bar, status_text, chart_placeholder, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.chart_placeholder = chart_placeholder
        self.total_epochs = total_epochs
        self.history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.history:
            if k in logs:
                self.history[k].append(logs[k])

        pct = min((epoch + 1) / self.total_epochs, 1.0)
        self.progress_bar.progress(pct)
        self.status_text.markdown(
            f"**Epoch {epoch + 1}/{self.total_epochs}** — "
            f"loss: `{logs.get('loss', 0):.5f}` · "
            f"val_loss: `{logs.get('val_loss', 0):.5f}` · "
            f"mae: `{logs.get('mae', 0):.5f}`"
        )

        hist_df = pd.DataFrame(
            {k: v for k, v in self.history.items() if len(v) > 0}
        )
        if not hist_df.empty:
            self.chart_placeholder.line_chart(hist_df)


def train_model(
    X_train,
    y_train,
    input_shape,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    patience=20,
    progress_callback=None,
):
    model = build_gru_model(input_shape)

    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)

    callbacks = [early_stop, reduce_lr]
    if progress_callback is not None:
        callbacks.append(progress_callback)

    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def evaluate_model(model, X_test, y_test, target_scaler):
    y_pred_scaled = model.predict(X_test, verbose=0)

    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2, "y_true": y_true, "y_pred": y_pred}


# ==========================================================
# SAVE / LOAD ARTIFACTS
# ==========================================================

def save_artifacts(model, feature_scaler, target_scaler, features, time_steps):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)
    joblib.dump(features, FEATURES_PATH)
    joblib.dump(time_steps, TIMESTEPS_PATH)


def artifacts_exist():
    return all(
        os.path.exists(p)
        for p in [MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH, FEATURES_PATH, TIMESTEPS_PATH]
    )


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_model(MODEL_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    time_steps = joblib.load(TIMESTEPS_PATH)
    return model, feature_scaler, target_scaler, features, time_steps


# ==========================================================
# RECURSIVE MULTI-DAY FORECAST
# ==========================================================

def recursive_forecast(
    df: pd.DataFrame,
    model,
    feature_scaler,
    target_scaler,
    features,
    time_steps,
    n_days: int,
    future_oil=None,
    future_gold=None,
    exog_trends=None,
):
    """
    Forecast USDIDR untuk n_days ke depan secara rekursif.

    future_oil, future_gold : list/array sepanjang n_days berisi asumsi
        nilai harian OIL & GOLD (jika None, nilai terakhir dipakai flat).
    exog_trends : dict opsional {"SP500": pct_per_day, "IHSG": ..., "VIX": ...,
        "CPI": ..., "BI_rate": ..., "US_rate": ...} — persentase perubahan
        harian yang diterapkan secara compounding dari nilai terakhir.
    """
    exog_trends = exog_trends or {}
    history = df[["Date"] + list(dict.fromkeys(
        ["USDIDR", "OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
    ))].copy().reset_index(drop=True)

    if future_oil is None:
        future_oil = [history["OIL"].iloc[-1]] * n_days
    if future_gold is None:
        future_gold = [history["GOLD"].iloc[-1]] * n_days

    last_date = history["Date"].iloc[-1]
    last_vals = {c: history[c].iloc[-1] for c in ["SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]}

    predictions = []
    dates = []

    for step in range(n_days):
        next_date = last_date + pd.Timedelta(days=step + 1)

        row = {
            "Date": next_date,
            "OIL": future_oil[step],
            "GOLD": future_gold[step],
        }
        for col in ["SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]:
            trend_pct = exog_trends.get(col, 0.0) / 100.0
            last_vals[col] = last_vals[col] * (1 + trend_pct)
            row[col] = last_vals[col]

        # Bangun fitur lag dari histori (termasuk hasil prediksi sebelumnya)
        def lag(col, n):
            return history[col].iloc[-n]

        feat_row = {
            "USDIDR": history["USDIDR"].iloc[-1],  # placeholder, akan dipakai sbg fitur window terakhir
            "OIL": row["OIL"],
            "GOLD": row["GOLD"],
            "SP500": row["SP500"],
            "IHSG": row["IHSG"],
            "VIX": row["VIX"],
            "CPI": row["CPI"],
            "BI_rate": row["BI_rate"],
            "US_rate": row["US_rate"],
            "USDIDR_Lag1": lag("USDIDR", 1),
            "USDIDR_Lag7": lag("USDIDR", 7) if len(history) >= 7 else lag("USDIDR", 1),
            "USDIDR_Lag30": lag("USDIDR", 30) if len(history) >= 30 else lag("USDIDR", 1),
            "OIL_Lag1": lag("OIL", 1),
            "OIL_Lag7": lag("OIL", 7) if len(history) >= 7 else lag("OIL", 1),
            "GOLD_Lag1": lag("GOLD", 1),
            "SP500_Lag1": lag("SP500", 1),
            "IHSG_Lag1": lag("IHSG", 1),
            "VIX_Lag1": lag("VIX", 1),
        }

        # window 30 hari terakhir (fitur), ganti baris terakhir agar konsisten
        # dengan asumsi hari berjalan, lalu prediksi hari berikutnya
        window_df = history.tail(time_steps).copy()
        # baris paling akhir window mewakili "hari ini"; kita append fitur
        # asumsi sbg baris baru dan ambil time_steps baris terakhir
        new_row_df = pd.DataFrame([{**feat_row, "Date": next_date}])
        window_full = pd.concat([window_df, new_row_df], ignore_index=True).tail(time_steps)

        X_window = feature_scaler.transform(window_full[features])
        X_window = X_window.reshape(1, time_steps, len(features))

        pred_scaled = model.predict(X_window, verbose=0)
        pred_value = target_scaler.inverse_transform(pred_scaled)[0, 0]

        predictions.append(pred_value)
        dates.append(next_date)

        # tambahkan hasil prediksi ke histori agar lag berikutnya konsisten
        new_actual_row = {
            "Date": next_date,
            "USDIDR": pred_value,
            "OIL": row["OIL"],
            "GOLD": row["GOLD"],
            "SP500": row["SP500"],
            "IHSG": row["IHSG"],
            "VIX": row["VIX"],
            "CPI": row["CPI"],
            "BI_rate": row["BI_rate"],
            "US_rate": row["US_rate"],
        }
        history = pd.concat([history, pd.DataFrame([new_actual_row])], ignore_index=True)

    return pd.DataFrame({"Date": dates, "USDIDR_Forecast": predictions})
