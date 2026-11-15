# Bab IV — Hasil dan Pembahasan (Bagian Baseline M1)

> **Status: DRAF** — disusun dari artefak eksperimen di `outputs/` (run `m1final`, `m1e1`, `m1e2`,
> `m1_tuning`, dan baseline lama `m1`). Bagian ini menjadi bahan Bab IV untuk porsi Anggota 2
> (baseline & ablasi). Angka dapat diperbarui bila M2–M5 dilatih ulang dengan epoch setara.

## 4.x.1 Setup eksperimen baseline

Model baseline **M1 (Late Fusion Scratch)** memakai Shared-Stem CNN pada log-mel `(4, 64, 800)`
dan encoder teks `subword`, digabung dengan *concatenation* lalu MLP menjadi 4 nilai gain.
Rincian arsitektur ada di Bab III.

Seluruh eksperimen M1 memakai split yang sama (80/20/50 lagu, *split by song*), label
multimodal yang sama, dan evaluasi pada tiga split: `validation`, `test_synthetic`, dan
`test_human` (200 prompt manusia). Metrik utama: MAE gain (dB), RMSE, R², *band MAE*,
*log-spectral distance* (LSD), `success_rate_1db` (MAE per sampel < 1 dB), dan
`instruction_adherence` (arah perubahan gain sesuai prompt).

Tabel 4.x.1 merangkum setelan tiap run. Perbedaan penting antar run adalah **panjang
training** dan **regularisasi**, bukan arsitektur: arsitektur M1 tidak diubah sama sekali
(452.358 parameter di semua run).

| Run | Epoch | Batch efektif | Dropout | Device | Versi torch |
|---|---|---|---|---|---|
| E0 (baseline lama) | 6 | 64 (batch langsung) | 0.15 | CPU | 2.14.1+cpu |
| E1 (uji akumulasi) | 6 | 8 x 8 (akumulasi) | 0.15 | CUDA + AMP | 2.15.0 |
| E2 | 30 | 8 x 8 | 0.15 | CUDA + AMP | 2.15.0 |
| **Final** | 50 | 8 x 8 | **0.0** | CUDA + AMP | 2.15.0 |

AMP dan akumulasi gradien dipakai agar batch mikro 8 tetap muat di VRAM 4 GB, dengan batch
efektif 64 seperti baseline lama.

## 4.x.2 Baseline non-learning

Sebagai titik acuan, dievaluasi tiga pembanding sederhana pada split yang sama (Tabel 4.x.2).

| Baseline | Split | MAE (dB) | RMSE (dB) | Band MAE | Succ<1dB | Adherence |
|---|---|---|---|---|---|---|
| No-op (0 dB) | validation | 3.9203 | 4.9197 | 2.1894 | 2.38% | 1.88% |
| No-op (0 dB) | test_human | 3.5678 | 4.5182 | 1.9261 | 2.50% | 2.50% |
| Random gain | validation | 4.7635 | 5.9204 | 2.9508 | 0.38% | 1.00% |
| Random gain | test_human | 4.6553 | 5.7948 | 2.8481 | 0.00% | 0.50% |
| DSP equal loudness | validation | 2.3342 | 2.7476 | 1.4604 | 6.12% | 5.75% |
| DSP equal loudness | test_human | 2.2412 | 2.6625 | 1.3911 | 5.50% | 5.00% |

Kedua baseline non-learning memiliki *success rate* sangat rendah (< 7%): tanpa informasi teks,
target keseimbangan relatif `r` tidak dapat diprediksi, sehingga MAE tetap > 2 dB.

## 4.x.3 Pengaruh gradient accumulation dan panjang training

**Validitas akumulasi gradien.** E1 (batch mikro 8, akumulasi 8) dibandingkan dengan E0
(batch 64 langsung), keduanya 6 epoch (Tabel 4.x.3). Selisih MAE validasi hanya
**+0.029 dB** (`0.8657` vs `0.8363`) dan test_human **+0.028 dB** (`0.9279` vs `0.9000`),
yaitu di dalam toleransi ±0.05 dB. Dengan demikian akumulasi gradien **setara secara
matematis** dengan batch besar, sehingga batch efektif 64 dapat dipertahankan di VRAM 4 GB.
Selisih kecil dapat diatribusikan pada perbedaan urutan acak dan *device* (CPU vs CUDA + AMP).

| Run | val MAE | Δ vs E0 | test_human MAE | Δ vs E0 |
|---|---|---|---|---|
| E0 (6 ep, batch 64, CPU) | 0.8363 ± 0.0505 | — | 0.9000 ± 0.0229 | — |
| E1 (6 ep, 8 x 8, CUDA+AMP) | 0.8657 ± 0.0217 | +0.0293 | 0.9279 ± 0.0326 | +0.0279 |

**Panjang training adalah faktor dominan.** Menambah epoch dari 6 ke 30 (E2) menurunkan MAE
validasi sebesar **−0.372 dB** dan test_human **−0.301 dB** — jauh lebih besar daripada efek
akumulasi. Pada E0/E1 semua run berhenti pada `best_epoch = 6` (epoch terakhir), artinya
model belum konvergen saat baseline lama dihentikan.

## 4.x.4 Hyperparameter tuning

Tuning kecil dijalankan pada seed 42 (Tabel 4.x.4), dengan E2 seed 42 sebagai referensi
(val 0.4328 / human 0.5770 / synthetic 0.4920). Kandidat mencakup panjang training,
dropout, dimensi tersembunyi, learning rate, dan weight decay.

| Kandidat | val MAE | test_human | test_synth | Δval vs E2 |
|---|---|---|---|---|
| **dropout 0.0 + 50 epoch** | **0.3287** | **0.4448** | **0.3439** | **−0.1041** |
| dropout 0.0 (30 ep) | 0.3556 | 0.4676 | 0.3721 | −0.0772 |
| 50 epoch (dropout 0.15) | 0.3774 | 0.5117 | 0.4179 | −0.0554 |
| hidden dim 512 | 0.4304 | 0.5713 | 0.4871 | −0.0024 |
| weight decay 1e-3 | 0.4363 | 0.5781 | 0.4937 | +0.0035 |
| hidden dim 128 | 0.4375 | 0.5911 | 0.5019 | +0.0047 |
| dropout 0.3 | 0.5152 | 0.6857 | 0.5815 | +0.0824 |
| lr 5e-4 | 0.5177 | 0.6283 | 0.5966 | +0.0849 |

Dua temuan:

1. **Menghapus dropout meningkatkan akurasi** (−0.077 dB val, −0.109 dB test_human). Sebaliknya,
   menaikkan dropout ke 0.3 atau menambah weight decay 1e-3 memperburuk hasil. Dengan kata lain
   M1 **over-regularized** pada dataset ini (3.200 sampel training, 452 ribu parameter).
2. **Training lebih lama masih membantu** (−0.055 dB), dan penggabungan kedua perubahan
   memberi perbaikan terbesar. Menurunkan learning rate ke 5e-4 justru memperburuk, karena
   memperlambat konvergensi pada jadwal *cosine* yang sudah dibatasi jumlah epoch.

Berdasarkan hasil ini, konfigurasi final M1 adalah **dropout 0.0, epochs 50, patience 8**.

## 4.x.5 Hasil final dan ablasi

Tabel 4.x.5 melaporkan hasil akhir M1 (mean ± std, 3 seed) beserta dua ablasi.

| Model | Split | MAE (dB) | RMSE (dB) | R² | Band MAE | LSD | Succ<1dB | Adherence |
|---|---|---|---|---|---|---|---|---|
| **M1 (multimodal)** | validation | **0.3252 ± 0.0025** | 0.4377 | 0.9921 | 0.1966 | 0.2365 | **99.66%** | **95.50%** |
| **M1 (multimodal)** | test_human | **0.4464 ± 0.0014** | 0.6385 | 0.9800 | 0.2558 | 0.3163 | **93.33%** | **88.17%** |
| **M1 (multimodal)** | test_synthetic | **0.3426 ± 0.0014** | 0.4574 | 0.9906 | 0.2034 | — | 99.40% | 94.98% |
| M1 audio-only | validation | 2.3171 ± 0.0142 | 2.7792 | 0.6808 | 1.3947 | 1.7062 | 5.29% | 3.79% |
| M1 audio-only | test_human | 2.2705 ± 0.0210 | 2.7198 | 0.6376 | 1.3637 | 1.6555 | 3.83% | 3.00% |
| M1 text-only | validation | 3.3303 ± 0.0005 | 4.2728 | 0.2457 | 1.9502 | 2.2847 | 4.91% | 4.41% |
| M1 text-only | test_human | 3.0796 ± 0.0184 | 3.8742 | 0.2647 | 1.9056 | 2.2458 | 4.83% | 4.33% |

Interpretasi:

- **Kedua modalitas diperlukan.** Multimodal (0.325 dB) jauh lebih baik daripada audio-only
  (2.317 dB) maupun text-only (3.330 dB). Selisih > 2 dB menunjukkan fusion teks–audio benar
  memberi informasi tambahan, bukan sekadar efek arsitektur.
- **Ablasi audio-only hampir menyamai baseline DSP** (2.317 dB vs 2.334 dB). Tanpa sinyal teks,
  model audio tidak dapat mengetahui target keseimbangan `r`, sehingga hanya mampu menebak
  loudness — sesuai desain label `g* = g0 + (L_ref + r − L)`.
- **Ablasi text-only hang** (MAE 3.33 dB, R² 0.25, success ≈ 5%, adherence ≈ 4%): model teks
  hanya menghasilkan gain kecil yang hampir konstan. Ini penting sebagai bukti **tidak ada
  kebocoran label lewat teks** pada setting ini (lihat §5.7 dokumen spesifikasi).
- Adherence multimodal ≈ 95% (validation) turun ke ≈ 88% pada prompt manusia, menunjukkan
  generalisasi bahasa yang baik namun tidak sempurna pada formulasi manusia.

## 4.x.6 Analisis error per stem dan per lagu

MAE per stem pada split validasi (rata-rata 3 seed):

| Stem | val MAE | test_human MAE |
|---|---|---|
| vocals | 0.246 | 0.443 |
| drums | 0.438 | 0.553 |
| bass | 0.338 | 0.391 |
| other | 0.278 | 0.398 |

*Drums* paling sulit diprediksi (MAE tertinggi) sementara *vocals* paling mudah. Pola ini
konsisten di kedua split dan masuk akal: kontribusi energi drum bersifat transien sehingga
loudness RMS-nya lebih fluktuatif.

Error per lagu pada validasi menunjukkan rentang 0.19–0.69 dB (rata-rata 3 seed):

| Lagi | val MAE |
|---|---|
| Bill Chudziak - Children Of No-one | 0.692 |
| The Scarlet Brand - Les Fleurs Du Mal | 0.472 |
| St Vitus - Word Gets Around | 0.465 |
| Swinging Steaks - Lost My Way | 0.404 |
| Lushlife - Toynbee Suite | 0.397 |

Lagu dengan error terbesar cenderung dinamis (transien tinggi / *sparse*), sesuai kesulitan
stems drum. Per-lagu lengkap ada pada `per_song` di `outputs/training/m1final_*_results.json`.

## 4.x.7 Perbandingan antar-model dan keterbatasan

Tabel 4.x.7 membandingkan M1 final dengan model lain **pada apa adanya artefak saat ini**.

| Model | Epoch artefak | `best_epoch` | val MAE | test_human MAE | Adherence (val) |
|---|---|---|---|---|---|
| M1 (final) | 50 | 50/50/46 | 0.3252 | 0.4464 | 95.5% |
| M1 (E0 lama) | 6 | 6/6/6 | 0.8363 | 0.9000 | 60.9% |
| M2 | 6 | 6/6/6 | 1.1982 | 1.2305 | 33.5% |
| M3 | 6 | 6/6/6 | 0.2739 | 0.4435 | 98.0% |
| M4 | 6 | 6/6/6 | 0.6265 | 0.8463 | 77.7% |
| M5 | 6 | 6/6/6 | 2.1879 | 2.1386 | 7.7% |

Keterbatasan penting yang harus dinyatakan secara eksplisit di Bab IV:

1. **Perbandingan belum apple-to-apple.** Hanya M1 yang dilatih sampai 50 epoch; M2–M5 masih
   pada 6 epoch dan semua `best_epoch = 6` (belum konvergen). M1 yang semula paling lemah
   (0.8363) kini menyamai M3 (0.2739 vs 0.3252 val) hanya karena dilatih lebih lama. Klaim
   "M1 mengalahkan M4" **tidak valid** sampai M4 dilatih ulang dengan epoch setara.
2. **Perbedaan device antar-run lama.** E0 dijalankan di CPU (torch 2.14.1), E1/E2/final di
   CUDA dengan AMP (torch 2.15.0). Karena E1 ≈ E0 pada 6 epoch, perpindahan device bukan
   penyebab utama lonjakan E2; namun untuk reproduksi ketat, seluruh run sebaiknya
   dijalankan pada device yang sama.
3. **Konfigurasi tidak sepenuhnya dari YAML.** `configs/*.yaml` pada repo ini hanya bersifat
   dokumentasi; konfigurasi efektif ditetapkan lewat argumen CLI. Untuk reproduksibilitas,
   Bab III sebaiknya memuat perintah CLI lengkap (lihat README §4).

## 4.x.8 Ringkasan temuan

1. **Akumulasi gradien valid**: batch 8 x akumulasi 8 ≈ batch 64 (selisih 0.03 dB), sehingga
   training batch besar dapat dijalankan pada VRAM 4 GB.
2. **Penyebab utama peningkatan M1 adalah panjang training dan regularisasi**, bukan perubahan
   arsitektur: 6 → 50 epoch dan dropout 0.15 → 0.0 memberi MAE val 0.8363 → 0.3252 dB
   (−61%) dan test_human 0.9000 → 0.4464 dB (−50%).
3. **Ablasi menegaskan kebutuhan multimodalitas**: audio-only 2.317 dB, text-only 3.330 dB,
   multimodal 0.325 dB. Text-only gagal secara meyakinkan → tidak ada *shortcut* lewat teks.
4. **Temuan lintas-peran (untuk didiskusikan dengan Anggota 3 & 4)**: seluruh baseline tim
   (M2–M5) terlatih hanya 6 epoch dan belum konvergen; perbandingan model pada Bab IV perlu
   diulang dengan jumlah epoch yang setara sebelum kesimpulan apa pun diambil.
