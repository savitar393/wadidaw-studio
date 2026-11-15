# M1 Baseline — Ringkasan Artefak Final (Anggota 2)

> Ringkasan ini dihasilkan dari run final `m1final` (3 seed) di `outputs/m1_final/`.
> Tabel deliverable utama ada di `outputs/m1_results.csv`.

## 1. Konfigurasi final

| Parameter | Nilai |
|---|---|
| Model | `m1` (`LateFusionScratchModel`, `wadidaw/models/fusion.py`) |
| Params (trainable) | 452.358 |
| Text backend | `subword` (encoder deterministik, offline) |
| `batch_size` x `grad_accum_steps` | 8 x 8 = 64 (efektif, setara baseline lama) |
| `epochs` / `patience` | 50 / 8 |
| `dropout` | 0.0 (override via CLI) |
| `lr` / `weight_decay` / `grad_clip` | 1e-3 / 1e-4 / 1.0 |
| Device / AMP | CUDA (`auto`) / aktif |
| Versi | python 3.12.13, torch 2.15.0, numpy 2.5.4 |
| Waktu rata-rata | ±351 s per run (50 epoch) |

Catatan: `configs/m1.yaml` sengaja dibiarkan pada versi aslinya (batch 16) dan **tidak** menjadi
sumber konfigurasi; seluruh setelan efektif diatur lewat CLI (lihat README §4).

## 2. Hasil final (mean ± std, 3 seed: 42/123/2026)

| Model | Split | MAE (dB) | RMSE (dB) | R² | Band MAE | LSD | Succ<1dB | Adherence |
|---|---|---|---|---|---|---|---|---|
| **m1** | validation | **0.3252 ± 0.0025** | 0.4377 | 0.9921 | 0.1966 | 0.2365 | 99.66% | 95.50% |
| **m1** | test_human | **0.4464 ± 0.0014** | 0.6385 | 0.9800 | 0.2558 | 0.3163 | 93.33% | 88.17% |
| **m1** | test_synthetic | **0.3426 ± 0.0014** | 0.4574 | 0.9906 | 0.2034 | — | 99.40% | 94.98% |
| m1_audio_only | validation | 2.3171 ± 0.0142 | 2.7792 | 0.6808 | 1.3947 | 1.7062 | 5.29% | 3.79% |
| m1_audio_only | test_human | 2.2705 ± 0.0210 | 2.7198 | 0.6376 | 1.3637 | 1.6555 | 3.83% | 3.00% |
| m1_text_only | validation | 3.3303 ± 0.0005 | 4.2728 | 0.2457 | 1.9502 | 2.2847 | 4.91% | 4.41% |
| m1_text_only | test_human | 3.0796 ± 0.0184 | 3.8742 | 0.2647 | 1.9056 | 2.2458 | 4.83% | 4.33% |

Baseline pembanding (non-learning, dari tabel yang sama):

| Baseline | Split | MAE (dB) | RMSE (dB) | Band MAE | Succ<1dB |
|---|---|---|---|---|---|
| no_op | validation | 3.9203 | 4.9197 | 2.1894 | 2.38% |
| no_op | test_human | 3.5678 | 4.5182 | 1.9261 | 2.50% |
| random | validation | 4.7635 | 5.9204 | 2.9508 | 0.38% |
| dsp_equal_loudness | validation | 2.3342 | 2.7476 | 1.4604 | 6.12% |
| dsp_equal_loudness | test_human | 2.2412 | 2.6625 | 1.3911 | 5.50% |

## 3. Progres M1 (val / test_human MAE)

| Run | Setelan | val | test_human | Catatan |
|---|---|---|---|---|
| E0 | 6 epoch, batch 64, CPU | 0.8363 ± 0.0505 | 0.9000 ± 0.0229 | baseline lama (artefak awal) |
| E1 | 6 epoch, 8 x accum 8, AMP | 0.8657 ± 0.0217 | 0.9279 ± 0.0326 | uji kesetaraan akumulasi (Δ ≈ +0.029 dB) |
| E2 | 30 epoch, 8 x accum 8, AMP | 0.4642 ± 0.0222 | 0.5994 ± 0.0163 | pengaruh utama = training lebih lama |
| final | 50 epoch, dropout 0.0 | **0.3252 ± 0.0025** | **0.4464 ± 0.0014** | konfigurasi final 3 seed |

## 4. Tuning kecil (seed 42, referensi E2 seed42 = 0.4328 / 0.5770 / 0.4920)

| Kandidat | val | test_human | test_synth |
|---|---|---|---|
| **dropout 0.0 + 50 epoch** | **0.3287** | **0.4448** | **0.3439** |
| dropout 0.0 (30 ep) | 0.3556 | 0.4676 | 0.3721 |
| 50 epoch (dropout default 0.15) | 0.3774 | 0.5117 | 0.4179 |
| hidden dim 512 | 0.4304 | 0.5713 | 0.4871 |
| weight decay 1e-3 | 0.4363 | 0.5781 | 0.4937 |
| hidden dim 128 | 0.4375 | 0.5911 | 0.5019 |
| dropout 0.3 | 0.5152 | 0.6857 | 0.5815 |
| lr 5e-4 | 0.5177 | 0.6283 | 0.5966 |

Papan peringkat lengkap: `outputs/m1_tuning_leaderboard.csv`.

## 5. Berkas

| Berkas | Isi |
|---|---|
| `outputs/m1_final/m1_results.csv` | tabel final (baseline + m1 + 2 ablasi) |
| `outputs/m1_final/per_seed_results.csv` | metrik per seed |
| `outputs/m1_final/main_results.csv` | tabel lengkap (mean ± std) |
| `outputs/m1_final/vs_e0.csv` | perbandingan final vs E0 |
| `outputs/m1_final/vs_e2.csv` | perbandingan final vs E2 |
| `outputs/m1_final/ablation_leaderboard.csv` | m1 vs audio-only vs text-only |
| `outputs/m1_results.csv` | **deliverable** (salinan tabel final) |
| `outputs/m1_results_e0.csv` | cadangan tabel E0 lama |
| `outputs/checkpoints/m1final_*_best.pt` | checkpoint final |
| `outputs/training/m1final_*_results.json` | detail evaluasi + kurva + per-lagu |
