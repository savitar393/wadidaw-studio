# Menjalankan eksperimen audio encoder di Google Colab Pro (L4 / A100)

Folder ini berisi **versi Google Colab** dari `notebooks/kaggle/1_audio_encoder_selection_kaggle.ipynb`.

- `1_audio_encoder_selection_colab.ipynb` — notebook siap upload/buka di Google Colab Pro.
- Seluruh logika eksperimen (head M3 structured, loss, `AUDIO_SEARCH_SPACE`, `TEXT_SEARCH_SPACE`, 5-fold CV, `effective_batch = 16`, seed, dan metrik) **identik** dengan versi Kaggle.

---

## 1. Struktur Google Drive

Notebook mengarah otomatis ke struktur berikut di `MyDrive`:

```text
ml_dabd_nlp/ml/
├── 1.Dataset/
│   ├── dataset/
│   │   ├── train/
│   │   └── test/
│   └── cache/
│       ├── logmel/         # 593 MB (1.500 .npy)
│       └── waveform16k/    # ±1,5 GB (otomatis dibuat Cell 2 lalu di-backup ke sini)
└── 2.Dokumentasi/
    └── m3/                 # Output persisten (member3/, zip, leaderboard, final/)
```

---

## 2. Optimasi GPU L4 / A100 (tetap sesuai kontrak fairness)

- **SSD lokal + RAM preload:** Cache `logmel` dan `waveform16k` disalin dari Drive ke SSD lokal `/content/wadidaw-studio/data/cache/` dan dimuat ke RAM saat RAM VM ≥ 24 GB (menghindari bottleneck I/O FUSE Drive).
- **TF32 + `bfloat16` AMP:** Aktif otomatis pada GPU L4 (24 GB) dan A100 (40/80 GB) tanpa overflow FP16 dan tanpa overhead `GradScaler`.
- **Physical batch adaptif (`effective_batch = 16` tetap terkunci):**
  - **A100 (≥ 36 GB VRAM):** `batch_size = 16 × grad_accum_steps = 1`
  - **L4 (≥ 20 GB VRAM):** `batch_size = 8 × grad_accum_steps = 2`
  - **T4 (< 20 GB VRAM):** `batch_size = 4 × grad_accum_steps = 4`
  - *(Khusus `cnn_scratch` yang memakai `BatchNorm1d`, tetap dikunci `4 × 4`.)*
  - Jika terjadi OOM, `run_one_with_retry` otomatis turun bertahap (`16 → 8 → 4 → 2`) dengan menjaga hasil kali tetap `16`.
- **DataLoader:** `num_workers = min(4, cpu_count)`, `pin_memory = True`, `persistent_workers = True`, `prefetch_factor = 2`.
- **Auto-sync ke Drive:** Hasil setiap fold/trial dan ZIP akhir di Cell 13 otomatis disalin ke `ml_dabd_nlp/ml/2.Dokumentasi/m3/`.

---

## 3. Cara Menjalankan (`MEMBER = 3` / `mert`)

1. Buka [colab.research.google.com](https://colab.research.google.com) → upload/buka `1_audio_encoder_selection_colab.ipynb`.
2. **Runtime → Change runtime type** → pilih **L4** (paling hemat CCU, ±3–5 CCU untuk Phase 1) atau **A100**.
3. Pastikan `MEMBER = 3` dan `AUTO_RUN = True` di **Cell 1**.
4. Jalankan seluruh sel berurutan (**Runtime → Run all**).
5. Setelah selesai dan hasil tersinkron ke `ml_dabd_nlp/ml/2.Dokumentasi/m3/`, klik **Runtime → Disconnect and delete runtime** agar CCU tidak terpakai saat idle.
6. Salin isi `member3/` ke `outputs/encoder_selection/member3/` di repo GitHub lalu commit & push.
