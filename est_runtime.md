Ran extract measured train_sec from existing logs
Analyzed designexperiment.md:L1-117
Ran analyze exact sec/epoch by encoder and GPU
Ran check dataset overhead and historical run metadata
Ran compute exact 57-run runtime estimates

Estimasi ini dihitung dari **log `train_sec` aktual** di folder `outputs/encoder_selection/` (bukan tebakan kasar):
- Log `ast`, `mert`, `beats` berasal dari **NVIDIA L4** pada 5-fold CV (64 train / 16 val), lalu diskalakan `1,25×` ke holdout (80 train / 20 val).
- Log `cnn_scratch` berasal dari **Tesla T4** pada holdout (80/20) secara langsung.
- Log `text_only` (IndoBERT) diturunkan dari selisih `bimodal - audio_only` di L4 (`~25–30 detik/epoch` ditambah overhead DataLoader).
- Karena `max_epochs = 5` dan `patience = 3`, mayoritas run akan berjalan **penuh 5 epoch** (hanya bisa berhenti di epoch 4 jika `best_epoch = 1`). Jadi angka di bawah memakai patokan 5 epoch penuh.

---

### 1. Estimasi Waktu per 1 Run (Holdout 80/20, Max 5 Epoch)

| Encoder | Mode | GPU Basis Log | Detik/Epoch (Holdout 80/20) | **Estimasi / Run (5 Epoch)** |
|---|---|---|---:|---:|
| `cnn_scratch` | `audio_only` | Tesla T4 (aktual) | ~74 detik | **~6 menit** *(L4: ~2,5 mnt)* |
| `cnn_scratch` | `bimodal` | Tesla T4 (aktual) | ~144 detik | **~12 menit** *(L4: ~4,5 mnt)* |
| `mert` | `audio_only` | NVIDIA L4 | ~313 detik | **~26 menit** |
| `mert` | `bimodal` | NVIDIA L4 | ~338 detik | **~28 menit** |
| `beats` | `audio_only` | NVIDIA L4 | ~401 detik | **~33,5 menit** |
| `beats` | `bimodal` | NVIDIA L4 | ~431 detik | **~36 menit** |
| `ast` | `audio_only` | NVIDIA L4 | ~489 detik | **~41 menit** |
| `ast` | `bimodal` | NVIDIA L4 | ~514 detik | **~43 menit** |
| `text_only` (IndoBERT) | `text_only` | NVIDIA L4 (turunan) | ~35 detik | **~3 menit** *(T4: ~7 mnt)* |

---

### 2. Rincian Total 57 Run (Serial vs Paralel 4 Anggota di L4)

| Fase | Jumlah Run | Rincian Beban | Akumulasi GPU-Jam (Serial) | **Waktu Nyata / Wall-Clock (Paralel)** |
|---|---:|---|---:|---:|
| **Fase 1:** Baseline Awal + Stage 1 + Stage 2 + Stage 3 | **37 run** | • M1 `cnn_scratch` (8 run, T4): ~1,2 jam<br>• M2 `ast` (8 run, L4): **~5,6 jam**<br>• M3 `mert` (8 run, L4): ~3,6 jam<br>• M4 `beats` (8 run, L4): ~4,6 jam<br>• `text_only` (5 run, L4): ~0,25 jam | **~15,3 jam** | **~5,6 jam** *(bottleneck di M2 `AST`)* |
| **Fase 2:** Stage 4 Text OFAT (Top-2) | **8 run** | 4 run × 2 encoder Top-2 (paralel 2 anggota):<br>• `MERT + BEATs`: ~1,9 + ~2,4 jam<br>• `AST + BEATs`: ~2,9 + ~2,4 jam | **~4,3 sampai 5,3 jam** | **~2,4 sampai 2,9 jam** |
| **Fase 3:** Ablation Akhir Test (4 model × 3 seed + eval test) | **12 run** | Dibagi 1 konfigurasi (3 seed) per anggota:<br>• Pemenang `MERT` / `BEATs` / `AST` | **~4,6 sampai 6,8 jam** | **~1,5 sampai 2,2 jam** *(jika dibagi 4 orang)* |
| **Modality Removal** *(inference-time)* | **0 run training** | 2 mode × 3 seed di checkpoint final | **~2 menit** | **~2 menit** |
| **TOTAL (57 Run)** | **57 run** | **M1 T4, M2–M4 L4** | **~25 sampai 27,5 GPU-jam** | **~9,5 sampai 10,7 jam** |

*(Tambahkan overhead non-training sekitar 15 sampai 20 menit per sesi untuk mount Drive, copy cache ke SSD lokal, dan sinkronisasi antar-fase).*

---

### 3. Ringkasan Cepat Berdasarkan Skenario GPU

- **Serial di 1 GPU NVIDIA L4:** **~25 sampai 27,5 jam** (tidak muat dalam 1 sesi Colab, butuh 3 sesi).
- **Paralel 4 anggota di NVIDIA L4 (T4 untuk M1):** **~10 sampai 11 jam** waktu nyata, dan setiap fase (5,6 jam → 2,9 jam → 2,2 jam) **muat nyaman dalam 1 sesi Colab Pro** tanpa kena timeout.
- **Paralel 4 anggota di NVIDIA A100** *(estimasi ~2× lebih cepat dari L4 dengan `batch_size=16×1`)*: **~4,5 sampai 5,5 jam** waktu nyata (**~12 sampai 14 GPU-jam**).

### Tips Penjadwalan supaya Tidak Molor
1. **Taruh `text_only` (Stage 2 + baseline teks = 5 run) di Member 1 (`cnn_scratch`) atau Member 3 (`mert`)**, bukan di Member 2 (`ast`), karena Member 2 sudah menjadi bottleneck terlama di Fase 1 (~5,6 jam).
2. **Di Fase 3 (12 run test final), bagi 1 konfigurasi (3 seed) per anggota.** Jika semua 12 run ditumpuk di 1 anggota pemenang, fase final saja bisa memakan ~5 sampai 7 jam sendiri; kalau dibagi 4 orang, selesai dalam ~2 jam.