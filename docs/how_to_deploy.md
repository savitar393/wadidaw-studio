# Panduan Deployment Lokal — WadiDAW Studio

Dokumen ini menjelaskan langkah-langkah menjalankan aplikasi **WadiDAW Studio (Text-Guided Automatic Gain Staging)** secara lokal menggunakan **Streamlit**.

## 1. Prasyarat Sistem

- Python `3.10+` (direkomendasikan `Python 3.12`)
- Berjalan di **CPU** maupun **GPU CUDA** (aplikasi memiliki *automatic CPU fallback* sehingga tidak memerlukan GPU diskrit saat demo).
- Ruang disk minimal `2 GB` untuk environment dan checkpoint model (atau `~30 GB` jika menyertakan dataset penuh `MUSDB18-HQ`).

## 2. Instalasi

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Menyiapkan Checkpoint Model

Jika belum ada checkpoint di `outputs/checkpoints/`, latih model utama **M3 (`StructuredTwoBranchModel`)** terlebih dahulu:

```bash
python -m scripts.build_dataset --workers 4
python -m scripts.train_models --models m3 --seeds 42 --epochs 20
```

Checkpoint terbaik akan tersimpan secara otomatis di `outputs/checkpoints/m3_seed42_best.pt`.
> **Catatan:** Meskipun belum ada file checkpoint, aplikasi Streamlit tetap dapat dijalankan menggunakan mode `DSP + Rule Parser (Fallback)`.

## 4. Menjalankan Aplikasi Streamlit

Dari direktori utama repositori (`wadidaw-studio/`), jalankan:

```bash
streamlit run app/streamlit_app.py
```

Buka URL lokal yang ditampilkan di terminal (umumnya `http://localhost:8501`).

## 5. Alur Penggunaan Aplikasi

1. **Pilih Model Checkpoint** di panel kiri (*sidebar*), misalnya `m3_seed42`.
2. **Pilih Sumber Audio 4 Stem:**
   - **Upload 4 File WAV:** Unggah masing-masing file `vocals.wav`, `drums.wav`, `bass.wav`, dan `other.wav`.
   - **Pilih Lagu Test MUSDB18-HQ:** Pilih salah satu lagu dari `data/raw/musdb18hq/test` untuk pengujian cepat.
3. **Masukkan Instruksi Prompt Bahasa Indonesia**, misalnya:
   - `"vokal paling menonjol, bass agak mundur, drum seimbang"`
   - `"buat drum lebih maju, bass sedikit lebih maju, instrumen pengiring lebih mundur"`
4. Klik tombol **Mix 4 Stem**.
5. Aplikasi akan menampilkan:
   - Nilai **Gain Prediksi (dB)** untuk keempat stem (`VOCALS`, `DRUMS`, `BASS`, `OTHER`) beserta *loudness* awalnya.
   - Metrik kualitas audio (**Peak Level dBFS** dan **Clipping Rate %**).
   - Pemutar audio pratinjau (*audio preview*) dan tombol **Download `final_mix.wav`** (44.1 kHz Stereo).
