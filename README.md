# WadiDAW Studio — Text-Guided Automatic Gain Staging

Sistem pencampuran musik (*gain staging* 4 stem) berbasis instruksi teks Bahasa Indonesia menggunakan pendekatan **Supervised Multimodal Regression** pada dataset **MUSDB18-HQ**.

---

## Ringkasan Arsitektur

- **Input:** 4 stem audio (`vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`) + prompt teks Bahasa Indonesia (contoh: `"vokal paling menonjol, bass agak mundur, drum seimbang"`).
- **Representasi Audio:** Log-Mel Spectrogram `(4, 64, 800)` pada 16 kHz mono (durasi excerpt 8 detik) dengan skala dBFS global yang mempertahankan informasi *loudness* absolut tiap stem.
- **Label Multimodal:**
  - $L_{\text{ref}} = \text{mean}(L_{\text{active}})$
  - $g_i^* = \text{clip}(L_{\text{ref}} + r_i - L_i, -18, 12)\text{ dB}$ (dengan *silence gating* `> -45 dBFS` sehingga stem hening di-mask `0 dB`).
- **Model yang Dibandingkan (`wadidaw/models/`):**
  - **M1 (`LateFusionScratchModel`):** Shared-Stem CNN (scratch) + Text Encoder $\rightarrow$ Concat + MLP $\rightarrow g^*$ (beserta ablasi `m1_audio_only` dan `m1_text_only`).
  - **M2 (`LateFusionPretrainedModel`):** Multi-Scale Spectral Backbone (frozen) + Text Encoder $\rightarrow$ Concat + MLP $\rightarrow g^*$.
  - **M3 (`StructuredTwoBranchModel` — Model Utama):** Audio CNN $\rightarrow \hat{L}$ (4), Text MLP $\rightarrow \hat{r}$ (4, $\sum \hat{r}_i = 0$), digabung secara analitik $\hat{g}_i = \hat{L}\_{\text{ref}} + \hat{r}_i - \hat{L}_i$ dengan *multi-task auxiliary loss*.
  - **M4 (`CrossAttentionGainModel`):** Token klausa teks sebagai *Queries* dan token spasi-waktu stem audio sebagai *Keys/Values* melalui *Multi-Head Cross-Attention*.
  - **M5 (`FiLMConditionedModel`):** *Feature-wise Linear Modulation* ($\gamma \cdot F + \beta$) dari embedding teks terhadap peta fitur CNN.
- **Rendering Akhir:** Menggunakan stem asli **44.1 kHz stereo** melalui [`render_stereo`](wadidaw/stereo_mixer.py).

---

## Struktur Direktori

```text
wadidaw-studio/
├── app/
│   └── streamlit_app.py             # Entrypoint aplikasi Streamlit
├── configs/
│   ├── m1.yaml                      # Konfigurasi M1 + ablasi
│   ├── m2.yaml                      # Konfigurasi M2
│   ├── m3.yaml                      # Konfigurasi M3 (Model Utama)
│   └── m4.yaml                      # Konfigurasi M4
├── data/
│   ├── raw/musdb18hq/               # Dataset MUSDB18-HQ (100 train, 50 test)
│   ├── splits/musdb18hq_v1.json     # Split 80 train / 20 val / 50 test (artist-disjoint)
│   ├── prompts/                     # Bank prompt & manifest sampel multimodal
│   └── cache/logmel/                # Cache log-mel (4, 64, 800) dalam float16 (.npy)
├── docs/
│   └── how_to_deploy.md             # Panduan deployment lokal Streamlit
├── outputs/
│   ├── checkpoints/                 # Bobot model terbaik (*_best.pt)
│   ├── training/                    # Log kurva training (.csv) dan detail evaluasi (.json)
│   ├── m1_results.csv               # Tabel hasil baseline & ablasi M1
│   └── main_results.csv             # Tabel hasil perbandingan utama (mean ± std)
├── scripts/
│   ├── create_song_splits.py        # Generator split manifest & bank prompt
│   ├── build_dataset.py             # Ekstraktor cache log-mel & sampel multimodal
│   ├── train_models.py              # Script training & evaluasi M1–M5 + baseline
│   ├── compare_m1.py                # Pembanding/papan peringkat run M1 (pairwise & --runs)
│   ├── test_data_pipeline.py        # Unit test pipeline data & prompt
│   ├── test_models_and_training.py  # Unit test model, loss, & training
│   └── test_stereo_mixer.py         # Unit test DSP stereo renderer
├── wadidaw/
│   ├── app/                         # Pipeline inferensi & UI Streamlit
│   ├── data/                        # Modul MUSDB18-HQ, fitur log-mel, prompt, Dataset
│   ├── models/                      # Implementasi M1, M2, M3, M4, M5
│   ├── training/                    # Trainer & evaluator (parameter, spektral, teks)
│   └── stereo_mixer.py              # Differentiable W3C stereo renderer
├── dokumentasi.md                   # Dokumen perencanaan & spesifikasi proyek
└── requirements.txt
```

---

## Cara Menjalankan

### 1. Persiapan Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Membuat Split Manifest & Cache Dataset MUSDB18-HQ
```bash
python -m scripts.create_song_splits
python -m scripts.build_dataset --workers 6
```

### 3. Menjalankan Unit Tests
```bash
python -m unittest scripts/test_stereo_mixer.py scripts/test_data_pipeline.py scripts/test_models_and_training.py
```

### 4. Melatih & Mengevaluasi Model (M1–M4 + Ablasi)

Setup default (proyek lama, batch 64):

```bash
python -m scripts.train_models --models m1 m1_audio_only m1_text_only m2 m3 m4 --seeds 42 123 2026 --epochs 25
```

> ⚠️ `train_models.py` menulis ulang `outputs/main_results.csv`, `outputs/per_seed_results.csv`, dan `outputs/m1_results.csv`. Gunakan `--out-dir` (sejak patch M1) supaya hasil lama tidak tertimpa.

#### 4a. Baseline M1 (Anggota 2) — batch kecil + AMP + gradient accumulation

Semua konfigurasi diatur via CLI; `configs/*.yaml` bersifat dokumentasi (bukan sumber konfigurasi) dan sengaja dibiarkan apa adanya — termasuk `configs/m1.yaml` yang masih menulis `batch_size: 16`. Setelan efektif untuk VRAM 4 GB diatur lewat CLI: `--batch-size 8 --grad-accum 8` agar *effective batch* tetap 64 seperti baseline lama.

```bash
# E1 — uji kesetaraan akumulasi (6 epoch)
python -m scripts.train_models --models m1 m1_audio_only m1_text_only \
  --batch-size 8 --grad-accum 8 --epochs 6 --patience 3 \
  --run-prefix m1e1 --out-dir outputs/m1_v2

# E2 — baseline utama (30 epoch)
python -m scripts.train_models --models m1 \
  --batch-size 8 --grad-accum 8 --epochs 30 --patience 5 \
  --run-prefix m1e2 --out-dir outputs/m1_v2
```

#### 4b. Tuning kecil M1 (1 seed per kandidat)

```bash
python -m scripts.train_models --models m1 --seeds 42 \
  --batch-size 8 --grad-accum 8 --epochs 50 --patience 8 \
  --run-prefix tune_ep50 --out-dir outputs/m1_tuning
# varian lain: --dropout 0.0 | --dropout 0.3 | --hidden-dim 128 | --hidden-dim 512
#              --lr 5e-4 | --weight-decay 1e-3   (epochs 30, patience 5)
```

> ℹ️ `train.py` menolak menimpa output yang sudah ada (`FileExistsError: Output already exists: ...`). Untuk mengulang prefix yang sama (`--run-prefix`), tambahkan `--overwrite`; kalau tidak, ganti prefix. `outputs/checkpoints/` dan `outputs/training/` selalu menampung hasil tiap prefix secara terpisah, sehingga baseline resmi (`m1_seed*`, `m1e1_*`, `m1e2_*`) aman.

#### 4c. Menjalankan final 3 seed + ablasi

```bash
python -m scripts.train_models --models m1 m1_audio_only m1_text_only \
  --seeds 42 123 2026 --batch-size 8 --grad-accum 8 --epochs 30 --patience 5 \
  --run-prefix m1final --out-dir outputs/m1_final
# outputs/m1_final/m1_results.csv  -> bisa dijadikan outputs/m1_results.csv (backup dulu yang lama)
```

#### 4d. Membandingkan run M1 (`scripts/compare_m1.py`)

```bash
# Pairwise (baseline lama vs kandidat), per-seed + summary mean±std
python -m scripts.compare_m1 --baseline "m1_seed{seed}" \
  --candidate "m1e2_m1_seed{seed}" --out outputs/m1_e2_vs_e0.csv

# Leaderboard banyak kandidat (mis. hasil tuning, 1 seed)
python -m scripts.compare_m1 --baseline "m1e2_m1_seed{seed}" --seeds 42 \
  --runs "tune_ep50_m1_seed{seed},tune_drop30_m1_seed{seed},tune_hid512_m1_seed{seed}" \
  --out outputs/m1_tuning_leaderboard.csv
```

Catatan: hasil artefak lama tetap tersimpan — `outputs/m1_v2/` (E1/E2), `outputs/m1_tuning/`, `outputs/m1_final/`, dan `outputs/m1_results_e0.csv` (cadangan tabel E0).

> **Pemetaan deliverable (brief Anggota 2 → repo):** `models/m1_late_fusion.py` ≡ `wadidaw/models/fusion.py` → `LateFusionScratchModel` (plus ablasi `m1_audio_only`/`m1_text_only`); `configs/m1.yaml`; `outputs/m1_results.csv`.

### 5. Menjalankan Aplikasi Deployment Lokal (Streamlit)
```bash
streamlit run app/streamlit_app.py
```
