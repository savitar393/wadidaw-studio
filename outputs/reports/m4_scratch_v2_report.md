# Laporan Training M4 Scratch v2

## 1. Ringkasan

| Item | Nilai |
|---|---|
| Model | M4 Cross-Attention Multimodal (Scratch) |
| Run prefix | `m4_scratch_v2` |
| Branch | `feat/m4-crossattention` |
| Python | 3.12.13 |
| PyTorch | 2.15.0+cu132 |
| CUDA | 13.2 |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| VRAM | 8.18 GB |
| Device training | `cuda` |
| Seeds | 42, 123, 2026 |
| Epoch maksimum | 30 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| AMP | Ya |
| Text encoder | `subword` |
| Audio input | Log-mel 16 kHz mono `(4, 64, 800)` |
| OpenL3 | Tidak digunakan |

## 2. Perintah Training

```bash
.venv/bin/python -m scripts.train_models \
  --models m4 \
  --seeds 42 123 2026 \
  --epochs 30 \
  --batch-size 16 \
  --device cuda \
  --run-prefix m4_scratch_v2
```

## 3. Hasil per Seed

| Seed | Best epoch | Train time (s) | Val MAE | Test human MAE | Test synthetic MAE | Test human Succ<1dB | Test human Adherence |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 123 | 29 | 165.75 | 0.3025 | 0.4595 | 0.3266 | 90.00% | 86.00% |
| 2026 | 29 | 161.72 | 0.2984 | 0.4468 | 0.3322 | 89.50% | 85.50% |
| 42 | 10 | 86.21 | 0.3524 | 0.5038 | 0.3967 | 90.50% | 87.50% |

## 4. Ringkasan Mean ± Std M4

| Split | MAE (dB) | RMSE (dB) | R² | Succ<1dB | Band MAE (dB) | Adherence |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.3178 ± 0.0245 | 0.4397 ± 0.0330 | 0.9919 ± 0.0012 | 99.3800 ± 0.7071% | 0.1930 ± 0.0118 | 95.5833 ± 0.7728% |
| test_human | 0.4700 ± 0.0244 | 0.6739 ± 0.0184 | 0.9777 ± 0.0012 | 90.0000 ± 0.4082% | 0.2711 ± 0.0097 | 86.3333 ± 0.8498% |
| test_synthetic | 0.3518 ± 0.0318 | 0.4719 ± 0.0378 | 0.9899 ± 0.0016 | 98.9833 ± 0.4871% | 0.2085 ± 0.0205 | 95.3667 ± 0.6860% |

## 5. Perbandingan M1 vs M4

| Split | Metrik | M1 | M4 | Selisih |
|---|---:|---:|---:|---:|
| validation | MAE (dB) | 0.8363 | 0.3178 | -0.5186 |
| validation | Succ<1dB (%) | 71.1233 | 99.3800 | +28.2567 |
| validation | Adherence (%) | 60.8800 | 95.5833 | +34.7033 |
| test_human | MAE (dB) | 0.9000 | 0.4700 | -0.4300 |
| test_human | Succ<1dB (%) | 65.8333 | 90.0000 | +24.1667 |
| test_human | Adherence (%) | 54.1667 | 86.3333 | +32.1667 |
| test_synthetic | MAE (dB) | 0.8849 | 0.3518 | -0.5331 |
| test_synthetic | Succ<1dB (%) | 66.5500 | 98.9833 | +32.4333 |
| test_synthetic | Adherence (%) | 56.7000 | 95.3667 | +38.6667 |

## 6. Catatan

- M4 scratch dilatih di GPU, bukan CPU.
- M4 tidak memakai OpenL3 atau IndoBERT; encoder teks masih `subword`.
- Perbandingan M1 vs M4 bersifat **system-level**, karena encoder audio dan mekanisme fusi berbeda.
- Seed 42 berhenti lebih awal pada epoch 15 (best epoch 10); seed 123 dan 2026 mencapai epoch 30.
- Checkpoint disimpan di `outputs/checkpoints/m4_scratch_v2_m4_seed*_best.pt`.
- Hasil JSON/CSV disimpan di `outputs/training/`.
- Unit test: 29 test lulus.
- Ada warning PyTorch tentang urutan `lr_scheduler.step()` dan `optimizer.step()`; tidak menggagalkan training.

## 7. File Hasil

```text
outputs/checkpoints/m4_scratch_v2_m4_seed42_best.pt
outputs/checkpoints/m4_scratch_v2_m4_seed123_best.pt
outputs/checkpoints/m4_scratch_v2_m4_seed2026_best.pt
outputs/training/m4_scratch_v2_m4_seed42_results.json
outputs/training/m4_scratch_v2_m4_seed123_results.json
outputs/training/m4_scratch_v2_m4_seed2026_results.json
outputs/main_results.csv
```
