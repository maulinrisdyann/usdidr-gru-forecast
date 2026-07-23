"""
data_utils.py
=============
Utility functions untuk memuat, membersihkan, dan melakukan feature
engineering pada dataset finansial USDIDR — mereplikasi persis tahapan
pra-pemrosesan pada notebook GRU aslinya.
"""

import glob
import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# KONSTANTA (harus identik dengan notebook training)
# ==========================================================

TARGET = "USDIDR"

DAILY_COLS = ["USDIDR", "OIL", "GOLD", "SP500", "IHSG", "VIX"]
MONTHLY_COLS = ["CPI", "BI_rate", "US_rate"]

FEATURES = [
    "USDIDR",
    "OIL",
    "GOLD",
    "SP500",
    "IHSG",
    "VIX",
    "CPI",
    "BI_rate",
    "US_rate",
    "USDIDR_Lag1",
    "USDIDR_Lag7",
    "USDIDR_Lag30",
    "OIL_Lag1",
    "OIL_Lag7",
    "GOLD_Lag1",
    "SP500_Lag1",
    "IHSG_Lag1",
    "VIX_Lag1",
]

DEFAULT_TIME_STEPS = 30
KAGGLE_DATASET = "raphaelnazareth/indonesia-financial-time-series-dataset-2010-2026"


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data(show_spinner=False)
def load_from_kagglehub():
    """Coba unduh dataset asli lewat kagglehub (butuh kredensial Kaggle)."""
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv = glob.glob(os.path.join(path, "*.csv"))[0]
    return pd.read_csv(csv)


@st.cache_data(show_spinner=False)
def load_from_upload(file_bytes: bytes):
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


def load_raw_data(uploaded_file=None):
    """
    Prioritas sumber data:
      1. File CSV yang diupload user
      2. Dataset Kaggle asli (jika kredensial tersedia di environment)
    Mengembalikan (df, sumber) atau (None, pesan_error)
    """
    if uploaded_file is not None:
        try:
            df = load_from_upload(uploaded_file.getvalue())
            return df, "upload"
        except Exception as e:
            return None, f"Gagal membaca file upload: {e}"

    try:
        df = load_from_kagglehub()
        return df, "kagglehub"
    except Exception as e:
        return None, (
            "Tidak ada dataset yang tersedia. Silakan upload file CSV pada "
            "sidebar (kolom minimal: Date, USDIDR, OIL, GOLD, SP500, IHSG, "
            f"VIX, CPI, BI_rate, US_rate).\n\nDetail: {e}"
        )


# ==========================================================
# DATA CLEANING (identik dengan notebook)
# ==========================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df[DAILY_COLS] = df[DAILY_COLS].interpolate()
    df[MONTHLY_COLS] = df[MONTHLY_COLS].ffill()

    df = df.dropna().reset_index(drop=True)

    # buang data USDIDR yang tidak wajar (sesuai notebook)
    df = df[df["USDIDR"] > 5000].copy()
    df.reset_index(drop=True, inplace=True)
    return df


# ==========================================================
# FEATURE ENGINEERING (identik dengan notebook)
# ==========================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["USDIDR_Lag1"] = df["USDIDR"].shift(1)
    df["USDIDR_Lag7"] = df["USDIDR"].shift(7)
    df["USDIDR_Lag30"] = df["USDIDR"].shift(30)

    df["USDIDR_MA7"] = df["USDIDR"].rolling(7).mean()
    df["USDIDR_MA30"] = df["USDIDR"].rolling(30).mean()
    df["USDIDR_STD7"] = df["USDIDR"].rolling(7).std()

    df["OIL_Lag1"] = df["OIL"].shift(1)
    df["OIL_Lag7"] = df["OIL"].shift(7)

    df["GOLD_Lag1"] = df["GOLD"].shift(1)
    df["IHSG_Lag1"] = df["IHSG"].shift(1)
    df["SP500_Lag1"] = df["SP500"].shift(1)
    df["VIX_Lag1"] = df["VIX"].shift(1)

    df = df.dropna().reset_index(drop=True)
    return df


def prepare_dataset(raw_df: pd.DataFrame):
    """Full pipeline: cleaning + feature engineering -> siap dipakai model."""
    df = clean_data(raw_df)
    df = engineer_features(df)
    return df


# ==========================================================
# SCALING + SEQUENCE CREATION
# ==========================================================

def scale_features(df: pd.DataFrame, features=FEATURES, target=TARGET):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X = feature_scaler.fit_transform(df[features])
    y = target_scaler.fit_transform(df[[target]])

    return X, y, feature_scaler, target_scaler


def create_sequences(X, y, time_steps=DEFAULT_TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i : i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def train_test_split_sequences(X_seq, y_seq, train_ratio=0.8):
    train_size = int(len(X_seq) * train_ratio)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    return X_train, X_test, y_train, y_test, train_size
