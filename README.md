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
```bash
python -m scripts.train_models --models m1 m1_audio_only m1_text_only m2 m3 m4 --seeds 42 123 2026 --epochs 25
```

### 5. Menjalankan Aplikasi Deployment Lokal (Streamlit)
```bash
streamlit run app/streamlit_app.py
```
