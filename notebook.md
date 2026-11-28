Analyzed README.md:L1-163
Analyzed README.md:L1-51
Ran summarize cells of existing Kaggle and Colab notebooks
Ran read setup, encoders, model, and packaging cells
Analyzed dataset.py:L1-427
Analyzed evaluate.py:L1-329
Analyzed output.txt:L140-900
Analyzed output.txt:L900-1010
Ran check notebooks directory layout and git status
Analyzed features.py:L1-356
Ran check git remote, branches, and prompt/cache files
Ran check v2 branch commits and prompt manifest schemas
Created build_unified_notebook.py
Edited build_unified_notebook.py
Edited build_unified_notebook.py
Edited build_unified_notebook.py
Edited build_unified_notebook.py
Edited build_unified_notebook.py
Edited build_unified_notebook.py
Edited build_unified_notebook.py
Edited build_unified_notebook.py
Ran generate notebook and syntax-check all code cells · Exited with code 1
Ran fix raw string quoting, build notebook, and AST-check all cells
Ran run full 57-run end-to-end simulation and PyTorch smoke test · Exited with code 1
Ran run end-to-end 57-run test with .venv Python · Exited with code 1
Ran execute full 57-run end-to-end workflow validation · Exited with code 1
Edited build_unified_notebook.py
Edited build_unified_notebook.py
Ran run full verification suite and check git status
Edited build_unified_notebook.py
Ran run full 5-part validation suite and verify clean workspace

Notebook terpadu sudah dibuat di [1_audio_encoder_experiment_unified.ipynb](file:///home/acer/dev/wadidaw-studio/notebooks/1_audio_encoder_experiment_unified.ipynb) (13 sel: 2 markdown panduan + 11 sel kode, runnable di **Kaggle GPU** maupun **Google Colab GPU**).

---

### Fitur Utama Notebook

1. **Sesuai Persis Desain Eksperimen (Total 57 Training Run + 0 Run untuk 2 Inference-Time Removal):**
   - **Ablation Awal & Baseline** (default `CONFIG`, val, seed 42): 1 `text_only` + 4 `audio_only` + 4 `bimodal` (9 run).
   - **Stage 1 (Audio Unimodal OFAT):** per encoder (`lr` -> `wd`) (+12 run baru).
   - **Stage 2 (Text Unimodal OFAT):** sekali oleh Member 1 (`lr_text` -> `text_hidden`) (+4 run baru).
   - **Stage 3 (Bimodal Audio OFAT):** per encoder, cabang teks ikut dilatih dengan parameter baseline (+12 run baru).
   - **Stage 4 (Bimodal Text OFAT):** otomatis mengambil **Top-2 Stage 3**, kunci audio Stage 3, OFAT teks (+8 run baru).
   - **Ablation Akhir Test (3 seed `42, 123, 2026`, data `test_human` + `test_synthetic`):**
     1. `Unimodal Audio` (best Stage 1)
     2. `Unimodal Text` (best Stage 2)
     3. `Bimodal Text Only` (Inference-time Audio removal pada checkpoint Bimodal Final, 0 training run)
     4. `Bimodal Audio Only` (Inference-time Text removal pada checkpoint Bimodal Final, 0 training run)
     5. `Bimodal Baseline` (encoder pemenang Stage 4, default parameter)
     6. `Bimodal Final` (Audio Stage 3 + Text Stage 4)
   - **Max 5 Epoch Ketat:** dikunci di `MAX_EPOCHS = 5` dan diverifikasi dengan `assert` di training loop maupun sel Parity Checks.

2. **Strict Model Caching:**
   - Kunci cache SHA-256 mencakup `(protocol_version, encoder, mode, seed, max_epochs, patience, all_hyperparams)` termasuk parameter terkunci (`dropout_audio`, `dropout_text`, `lr_head`, `wd_head`, `wd_text`, `d_model`, `effective_batch`, `aux_lambda`).
   - Titik baseline dari Ablation Awal otomatis di-*reuse* oleh Stage 1, 2, dan 3; hasil terbaik Stage 3 otomatis di-*reuse* oleh Stage 4. Tidak bertabrakan antar-mode, antar-seed, maupun dengan output lama 7 epoch.

3. **Auto-Setup Kaggle & Colab (Dataset `satriamanggala/musdb18hq` & `deathstrawberry/waveform16k`):**
   - **Kaggle:** auto-deteksi dari `/kaggle/input` (mendukung struktur folder `train/validation/test`, struktur flat, maupun `.zip`) dengan fallback `kagglehub`.
   - **Colab:** auto-mount Google Drive (opsional, untuk sinkronisasi hasil tim) + auto-download `deathstrawberry/waveform16k` via `kagglehub` tanpa harus mengunduh 31 GB WAV `musdb18hq` apabila cache 1.500 excerpt sudah lengkap.
   - Metrik spektral `stem_mel_power` `(4, 64)` dihitung otomatis dari `waveform16k` (atau dari cache `logmel` jika ada) dan disimpan ke file ringan `stem_mel_power_16k.npz` (~1,5 MB) sehingga tidak wajib mengunggah dataset cache log-mel 593 MB terpisah.
   - Optimasi GPU otomatis (`bfloat16` + `batch_size=8×2` di L4, `16×1` di A100, `4×4` di T4/P100; `effective_batch=16` selalu terkunci, AMP FP16 dimatikan khusus BEATs di T4).

4. **Eksekusi Paralel 4 Anggota (Cukup Ubah `MEMBER` dan `PHASE` di Cell 1):**
   - **Phase 1 (`PHASE="phase1"`):** semua 4 anggota jalan paralel (`MEMBER=1` mengerjakan `cnn_scratch` + `text_only`, `MEMBER=2` `ast`, `MEMBER=3` `mert`, `MEMBER=4` `beats`).
   - **Phase 2 (`PHASE="phase2"`):** 2 anggota pemilik Top-2 Stage 3 jalan paralel (anggota non-Top-2 otomatis dilewati).
   - **Phase 3 (`PHASE="phase3"`):** 4 anggota jalan paralel, masing-masing 1 konfigurasi test × 3 seed (`M1`: Unimodal Audio, `M2`: Unimodal Text, `M3`: Bimodal Baseline, `M4`: Bimodal Final + 2 Inference-Time Removal). Bisa juga pakai `PHASE="auto"` atau `PHASE="report_only"`.

5. **Output Mudah Diunduh + Visualisasi Analisis:**
   - **4 Tabel CSV (`reports/`):** `table1_baseline_ablation.csv`, `table2_ofat_all_stages.csv`, `table3_stage_winners_val.csv`, `table4_final_test_ablation.csv`.
   - **4 Grafik 180 DPI (`figures/`):** perbandingan Baseline vs Tuned (validation), kurva sensitivitas OFAT Stage 1–4, bar chart Ablation Akhir Test (`mean ± std` untuk `test_human` dan `test_synthetic`), serta breakdown error per-stem dan kurva konvergensi 5 epoch.
   - **2 ZIP Terpisah** di `/kaggle/working/` (Kaggle) atau `/content/` + Google Drive (Colab):
     - `wadidaw_results_member<M>_<phase>.zip` (ringan tanpa `.pt`, langsung siap diunduh dan di-merge otomatis antar anggota via `restore_and_merge_team_outputs`).
     - `wadidaw_checkpoints_member<M>.zip` (khusus checkpoint `.pt` final).

---

### Hasil Verifikasi yang Dijalankan
- **Syntax check:** seluruh 11 sel kode lolos `ast.parse`.
- **PyTorch model smoke test (`EncoderProbeModel`):** forward pass `bimodal`, `audio_only` (tanpa alokasi cabang teks), `text_only` (tanpa alokasi audio encoder), `build_param_groups`, serta kesetaraan matematis *inference-time modality removal* ($g_{\text{bimodal}} = \text{clip}(g_{\text{no\_audio}} + g_{\text{no\_text}})$) lolos.
- **Deteksi dataset:** auto-deteksi `waveform16k` (struktur `train/` maupun flat) dibedakan secara akurat dari cache `logmel`.
- **Simulasi end-to-end 4 anggota × 3 fase (mocked training loop):** menghasilkan tepat **57 training run** (`37 + 8 + 12`) + **6 evaluasi inference-time removal** (0 training tambahan), re-run `PHASE="auto"` menghasilkan **0 run baru (100% cache hit)**, dan seluruh 4 tabel CSV, 4 grafik PNG, Parity Checks (Cell 10), ZIP ringan/checkpoint (Cell 11), serta auto-merge ZIP antar anggota lolos tanpa error.
- **Catatan:** pelatihan GPU penuh pada dataset `waveform16k` + checkpoint HuggingFace/BEATs belum dijalankan di mesin lokal (hanya simulasi end-to-end dan unit test forward PyTorch CPU); eksekusi training penuh dilakukan di Kaggle/Colab GPU.