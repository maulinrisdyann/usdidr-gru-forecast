# 💱 USDIDR GRU Forecast — Dashboard Streamlit

Dashboard interaktif untuk forecasting nilai tukar USD/IDR menggunakan model
GRU (Gated Recurrent Unit), dibangun berdasarkan pipeline notebook training
asli (`GRU2.ipynb`).

## 📁 Struktur Proyek

```
usdidr_gru_app/
├── Beranda.py                     # Halaman utama (entry point)
├── pages/
│   ├── 1_📊_Analisis_Data.py       # Grafik historis GOLD, OIL, dll
│   ├── 2_🧠_Training_Model.py      # Tahapan training + evaluasi
│   ├── 3_🔮_Prediksi_Forecast.py   # Forecast N hari ke depan
│   └── 4_🎮_Simulasi_Interaktif.py # What-if scenario simulator
├── utils/
│   ├── data_utils.py               # Cleaning & feature engineering
│   └── model_utils.py              # Build/train/evaluate/forecast GRU
├── model/                          # Artefak model tersimpan (otomatis dibuat)
├── requirements.txt
└── .streamlit/config.toml
```

## 🚀 Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run Beranda.py
```

## ☁️ Deploy ke Streamlit Community Cloud

1. Push folder ini ke repository GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io), pilih repo, dan set
   **Main file path** ke `Beranda.py`.
3. Klik **Deploy**.

## 📊 Sumber Data

Aplikasi ini mencoba dua sumber data (lihat sidebar halaman **Beranda**):

1. **Upload CSV manual** (disarankan untuk deployment publik) — kolom wajib:
   `Date, USDIDR, OIL, GOLD, SP500, IHSG, VIX, CPI, BI_rate, US_rate`
2. **Kaggle** (`raphaelnazareth/indonesia-financial-time-series-dataset-2010-2026`)
   via `kagglehub` — hanya berfungsi jika kredensial Kaggle (`kaggle.json` /
   environment variable `KAGGLE_USERNAME` & `KAGGLE_KEY`) tersedia di server.

Karena Streamlit Cloud biasanya tidak memiliki kredensial Kaggle, **upload CSV
manual adalah cara paling andal** untuk menjalankan dashboard ini setelah
deploy.

## 🧠 Alur Kerja

1. **Beranda** — muat data, lihat ringkasan & arsitektur pipeline.
2. **Analisis Data** — eksplorasi tren historis GOLD, OIL, SP500, IHSG, VIX,
   korelasi antar variabel, dan statistik deskriptif.
3. **Training Model** — jalankan training GRU langsung dari browser (dengan
   grafik loss live per epoch), atau muat model yang sudah tersimpan di folder
   `model/`. Hasil training otomatis disimpan dan bisa dipakai ulang tanpa
   perlu training ulang.
4. **Prediksi & Forecast** — input jumlah hari ke depan, atur asumsi harga
   GOLD/OIL (flat, tren %, atau edit manual per hari), lalu jalankan forecast
   rekursif multi-hari untuk USDIDR.
5. **Simulasi Interaktif** — geser slider kondisi pasar (GOLD, OIL, SP500,
   IHSG, VIX, suku bunga BI) dan lihat reaksi prediksi USDIDR secara instan,
   lengkap dengan gauge chart dan konverter mata uang cepat.

## ⚠️ Catatan Penting

- Forecast multi-hari bersifat **rekursif**: prediksi hari ke-N dipakai
  sebagai bagian input untuk memprediksi hari ke-N+1, sehingga error dapat
  terakumulasi pada horizon yang panjang.
- Nilai GOLD, OIL, dan variabel makro lain pada halaman forecast adalah
  **asumsi skenario** yang diinput pengguna, bukan data real-time aktual.
- Untuk data real-time, integrasikan API harga (mis. Yahoo Finance, Alpha
  Vantage) pada `utils/data_utils.py`.
