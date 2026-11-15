# Dokumentasi Proyek Machine Learning

## Text-Guided Automatic Gain Staging pada Pencampuran Musik Berbasis Multimodal Audio–Teks

> **Status:** Dokumen desain dan ringkasan keputusan  
> **Tanggal:** 6 November 2026  
> **Dataset:** MUSDB18-HQ  
> **Pendekatan utama:** Supervised multimodal regression  
> **Status RL:** SAC tidak digunakan sebagai model inti

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)  
2. [Latar Belakang dan Requirement](#2-latar-belakang-dan-requirement)  
3. [Definisi Task](#3-definisi-task)  
4. [Dataset dan Preprocessing](#4-dataset-dan-preprocessing)  
5. [Labeling Multimodal](#5-labeling-multimodal)  
6. [Model yang Direkomendasikan](#6-model-yang-direkomendasikan)  
7. [Training dan Inference](#7-training-dan-inference)  
8. [Evaluasi](#8-evaluasi)  
9. [Deployment](#9-deployment)  
10. [Pembagian Tugas 4 Orang](#10-pembagian-tugas-4-orang)  
11. [Timeline](#11-timeline)  
12. [Risiko dan Mitigasi](#12-risiko-dan-mitigasi)  
13. [Mapping Laporan](#13-mapping-laporan)  
14. [Checklist](#14-checklist)  
15. [Lampiran](#15-lampiran)

---

# 1\. Ringkasan Eksekutif

Proyek ini mengubah fokus dari **reinforcement learning untuk gain staging** menjadi:

> **Sistem pencampuran musik berbasis instruksi teks menggunakan model multimodal audio–teks.**

Input sistem:

```
4 stem audio + prompt Bahasa Indonesia
```

Output sistem:

```
4 nilai gain (dB) + 1 stereo mix
```

Empat stem yang digunakan:

```
vocals
drums
bass
other
```

Keputusan penting:

- **RL/SAC tidak dipakai sebagai model inti.**  
- Model utama adalah **supervised multimodal regression**.  
- Label dibuat dengan pendekatan **audio-dependent**:  
  - prompt → `r` (target keseimbangan),  
  - audio → `L` (loudness saat ini),  
  - `r + L` → `g*` (gain target).  
- Model minimal yang dibandingkan: **3 model Deep Learning**.  
- Rekomendasi: **4 model** agar memenuhi bonus \>3 model.  
- Deployment: **Streamlit lokal**.  
- Data utama: **MUSDB18-HQ**.

---

# 2\. Latar Belakang dan Requirement

## 2.1 Requirement utama

| Requirement | Isi |
| :---- | :---- |
| **Data multimodal** | Audio \+ teks |
| **Non-tabular** | Tidak boleh hanya memakai data tabular |
| **Deep Learning** | Minimal 3 model DL dibandingkan |
| **Deployment** | Minimal local deployment |
| **Laporan** | Format IEEE, Bahasa Indonesia |
| **Referensi** | Minimal 10 paper 5 tahun terakhir \+ 2 buku |
| **Notebook** | Runnable |
| **Dokumentasi deploy** | README, requirements, cara menjalankan |

## 2.2 Gap analysis kondisi awal

| Requirement | Kondisi awal | Status | Perbaikan |
| :---- | :---- | :---- | :---- |
| Multimodal | Audio saja | ❌ | Tambahkan prompt teks |
| Non-tabular | 101-dim fitur | ⚠️ | Model utama pakai log-mel |
| 3 model DL | SAC/MLP/XGBoost | ❌ | Buat 3–4 model supervised |
| Perbandingan adil | SAC vs no-op/random | ❌ | Semua model supervised, split sama |
| Deployment | Baru script | ❌ | Streamlit |
| Dataset | MUSDB audio | ⚠️ | Tambah teks sintetis \+ human test |
| SDG | Baru tema | ⚠️ | Bukti kegiatan nyata |
| Laporan | Belum | ⚠️ | Susun Bab I–V |

## 2.3 Scope yang dikunci

### Termasuk

- 4 stem: vocals, drums, bass, other.  
- Text-guided gain staging.  
- Output 4 gain dB.  
- Supervised multimodal regression.  
- Log-mel sebagai representasi utama.  
- Streamlit deployment.  
- Evaluasi kuantitatif \+ listening test kecil.

### Tidak termasuk

- EQ multi-band.  
- Low-pass/high-pass filter.  
- Compression, reverb, saturation.  
- Panning.  
- Piano-specific stem.  
- Full fine-tune MERT/CLAP/AST.  
- RL/SAC sebagai model inti.

## 2.4 SDG

### SDG 4 — Quality Education

- Tool untuk mahasiswa/musisi pemula belajar mixing.  
- Dokumentasi terbuka sebagai materi belajar.  
- User study kecil untuk mengevaluasi kegunaan.

### SDG 17 — Partnerships

- Kode dan dokumentasi open-source.  
- Reproducible experiment.  
- Kolaborasi dengan komunitas musik/kampus.

> Hindari klaim SDG tanpa bukti. Tulis aktivitas konkretnya di laporan.

---

# 3\. Definisi Task

## 3.1 Input dan output

### Input user

```json
{
  "stems": {
    "vocals": "vocals.wav",
    "drums": "drums.wav",
    "bass": "bass.wav",
    "other": "other.wav"
  },
  "prompt": "vokal lebih maju, bass agak mundur, drum seimbang"
}
```

### Output sistem

```json
{
  "gains_db": {
    "vocals": 4.75,
    "drums": -4.25,
    "bass": -3.25,
    "other": 2.75
  },
  "mix": "final_mix.wav",
  "metrics": {
    "clipping_rate": 0.0,
    "true_peak_dbtp": -1.0
  }
}
```

## 3.2 Formulasi supervised

```
(audio, prompt) → g*
```

Dengan auxiliary target:

```
audio  → L
prompt → r
```

Model dapat memprediksi `g*` langsung, atau memprediksi `L` dan `r` lalu menggabungkannya dengan rumus.

## 3.3 Mengapa bukan RL

| Aspek | Supervised | SAC |
| :---- | :---- | :---- |
| Data | pasangan audio–teks–gain | environment \+ reward |
| Training | forward/backward | interaksi \+ replay buffer |
| Stabilitas | lebih stabil | lebih sensitif |
| Sample efficiency | tinggi | rendah |
| Text conditioning | natural | rumit |
| Evaluasi | MAE/RMSE | return/success |
| Deployment | mudah | mudah, training sulit |

Alasan SAC di-drop:

- Task lebih cocok sebagai regresi satu langkah.  
- Target gain sudah tersedia dari simulasi.  
- Text conditioning lebih natural dengan supervised.  
- Replay buffer dan reward design menambah kompleksitas.  
- Perbandingan SAC vs supervised tidak apples-to-apples.  
- Hasil awal SAC belum meyakinkan.

SAC boleh tetap ada sebagai **future work** atau eksperimen tambahan, bukan model inti.

---

# 4\. Dataset dan Preprocessing

## 4.1 MUSDB18-HQ

Karakteristik:

| Aspek | Nilai |
| :---- | :---- |
| Jumlah lagu train | 100 |
| Jumlah lagu test | 50 |
| Stem | vocals, drums, bass, other |
| Sample rate | 44.1 kHz |
| Channel | Stereo |
| Format | WAV 16-bit |
| Total file | 750 WAV |
| Durasi total | ±9.82 jam |
| Durasi minimum | 13.09 detik |
| Durasi maksimum | 628.56 detik |

Struktur:

```
data/raw/musdb18hq/
├── train/
│   ├── Artist - Song/
│   │   ├── mixture.wav
│   │   ├── vocals.wav
│   │   ├── drums.wav
│   │   ├── bass.wav
│   │   └── other.wav
└── test/
    └── ...
```

Catatan:

- `mixture ≈ jumlah semua stem`.  
- Dataset tidak menyediakan label gain, teks, atau parameter mixing.  
- Semua label harus dibuat sendiri.  
- Cek lisensi MUSDB sebelum deployment publik.

## 4.2 Split

```
train/       80 lagu dari folder train MUSDB
validation/  20 lagu dari folder train MUSDB
test/        50 lagu resmi MUSDB
```

Aturan:

- Split **by song**, bukan by excerpt.  
- Jangan ada lagu yang sama di train dan test.  
- Prompt template juga dipisah antara train dan test.  
- Test set hanya dipakai sekali untuk evaluasi final.c

Manifest yang disarankan:

```
data/splits/musdb18hq_v1.json
```

## 4.3 Preprocessing audio

```
44.1 kHz stereo
   ↓
16 kHz mono
   ↓
potong 8 detik
   ↓
log-mel
```

Konfigurasi awal:

```
sample_rate : 16000
n_fft       : 1024
hop_length  : 160
n_mels      : 64
duration    : 8 detik
shape       : (4, 64, 800)
```

- `4` \= jumlah stem.  
- `64` \= mel bin.  
- `800` \= frame waktu.  
- Model bekerja di 16 kHz mono.  
- Render final memakai stem asli 44.1 kHz stereo.

## 4.4 Cache dan disk

- `/mnt/shared` hanya memiliki ±26 GB free.  
- Jangan simpan cache besar di `/mnt/shared`.  
- Simpan cache di `/home/ilydien` atau `/`.  
- Gunakan format `.npy`, `.npz`, atau HDF5.  
- Gunakan `float16` untuk cache embedding jika perlu.  
- Jangan load seluruh dataset ke RAM.

## 4.5 Dataset prompt

Satu excerpt dapat memiliki beberapa prompt:

```
1 excerpt × 4 prompt paraphrases
```

Target jumlah:

```
100 lagu × 10 excerpt × 4 prompt = 4.000 pasangan audio–teks
```

Tambahan:

```
200 prompt human-written untuk test
```

Prompt manusia dipakai untuk mengecek generalisasi, bukan untuk training.

## 4.6 Schema dataset

Contoh satu sampel:

```json
{
  "id": "train_0001_ex03_p02",
  "song": "Artist - Song",
  "split": "train",
  "excerpt_start_sec": 24.0,
  "duration_sec": 8.0,
  "audio_path": "cache/logmel/train_0001_ex03.npy",
  "g0_db": [0.0, 0.0, 0.0, 0.0],
  "loudness_db": [-18.0, -12.0, -15.0, -20.0],
  "balance_target": [3.0, 0.0, -2.0, -1.0],
  "gains_db": [4.75, -4.25, -3.25, 2.75],
  "prompt": "vokal paling menonjol, bass agak mundur, drum seimbang",
  "prompt_template": "vocal_forward_v1"
}
```

Catatan:

- `loudness_db`, `balance_target`, dan `gains_db` adalah label/auxiliary target.  
- Saat inference, hanya `audio` dan `prompt` yang diberikan.  
- Jangan masukkan `balance_target`, `gains_db`, atau target bands ke input model.

---

# 5\. Labeling Multimodal

## 5.1 Definisi variabel

| Simbol | Arti | Satuan |
| :---- | :---- | :---- |
| `g0_i` | Gain awal stem i | dB |
| `L_i` | Loudness stem i setelah gain `g0` | dBFS/LUFS |
| `r_i` | Target keseimbangan relatif dari prompt | dB |
| `L_ref` | Referensi loudness | dB |
| `g*_i` | Gain target untuk stem i | dB |

## 5.2 Rumus label baru

```
L_ref = mean(L_1, L_2, L_3, L_4)

g*_i = g0_i + (L_ref + r_i - L_i)
```

Dengan syarat:

```
sum(r_i) = 0
```

Jika `r` belum nol:

```
r_i = r_i - mean(r)
```

## 5.3 Mengapa perlu `L_ref`

Tanpa `L_ref`:

```
g*_i = g0_i + (r_i - L_i)
```

bisa membuat semua stem naik/turun bersamaan. Itu bukan mixing, tapi mengubah volume keseluruhan.

Dengan `L_ref`, rata-rata loudness dipertahankan, hanya keseimbangannya yang berubah.

## 5.4 Contoh perhitungan

Diketahui:

```
g0 = [0, 0, 0, 0] dB

L = [-18, -12, -15, -20] dB
# vocals, drums, bass, other

r = [+3, 0, -2, -1]
# vokal paling menonjol
# bass paling mundur
```

Hitung:

```
L_ref = mean(L)
      = (-18 - 12 - 15 - 20) / 4
      = -16.25 dB
```

Maka:

```
g*_vocals = 0 + (-16.25) + 3  - (-18) = 4.75 dB
g*_drums  = 0 + (-16.25) + 0  - (-12) = -4.25 dB
g*_bass   = 0 + (-16.25) + (-2) - (-15) = -3.25 dB
g*_other  = 0 + (-16.25) + (-1) - (-20) = 2.75 dB
```

Hasil:

```
g* = [4.75, -4.25, -3.25, 2.75]
```

Loudness akhir:

```
L*_vocals = -13.25 dB
L*_drums  = -16.25 dB
L*_bass   = -18.25 dB
L*_other  = -17.25 dB
```

Rata-rata tetap `-16.25 dB`, tetapi vokal paling menonjol dan bass paling mundur.

## 5.5 Prompt → r

Contoh mapping intensitas:

| Kata dalam prompt | Efek | Nilai `r` |
| :---- | :---- | ----: |
| `seimbang` | netral | 0 dB |
| `sedikit` / `agak` | kecil | ±2 dB |
| `lebih` | sedang | ±3 dB |
| `paling` / `sangat` | besar | ±5 dB |
| `mundur` | turun | \-2 sampai \-5 dB |
| `tonjolkan` / `maju` | naik | \+2 sampai \+5 dB |
| `jangan terlalu` | turun | \-2 dB |

Contoh:

```
"vokal paling menonjol, bass agak mundur, drum seimbang"
→ r = [+5, 0, -2, -3]
→ normalisasi mean
```

Catatan:

- Mapping ini adalah desain awal.  
- Nilai dapat disesuaikan setelah pilot study.  
- Untuk test manusia, `r` tidak diberikan ke model; model hanya melihat prompt.

## 5.6 Pseudocode pembuatan label

```py
for song in train_songs:
    for excerpt in sample_excerpts(song, n=10):
        g0 = random.uniform(-3, 3, size=4)

        stems = load_stems(song, excerpt)
        stems_g0 = apply_gain(stems, g0)

        L = measure_loudness(stems_g0)          # 4 nilai dB
        r = sample_balance_target()             # 4 nilai, sum = 0

        L_ref = mean(L)
        g_star = g0 + (L_ref + r - L)
        g_star = clip(g_star, -18, 12)

        if out_of_range(g_star):
            continue

        for prompt in generate_prompts(r, n=4):
            save_sample(
                audio=stems,
                g0=g0,
                L=L,
                r=r,
                g_star=g_star,
                prompt=prompt
            )
```

## 5.7 Anti-leakage

### Label leakage

Jangan:

```
❌ prompt → g* langsung
❌ prompt berisi angka gain final
❌ target bands dimasukkan ke input model
❌ r atau L dimasukkan ke input jika dihitung dari label
```

Lakukan:

```
✅ prompt → r
✅ audio → L
✅ r + L → g*
✅ input model hanya audio + prompt + current gain
```

### Data leakage

```
✅ split by song
✅ split by prompt template
✅ test dengan 200 prompt manusia
✅ test set hanya dipakai sekali
✅ normalisasi statistik hanya dari train
```

### Shortcut learning

Jika model text-only menang, berarti:

- task terlalu mudah,  
- prompt membocorkan jawaban,  
- atau audio tidak diperlukan.

Itu harus dilaporkan sebagai temuan, bukan disembunyikan.

---

# 6\. Model yang Direkomendasikan

## 6.1 Audio encoder

| Kategori | Model | Input | Output | Pretrained | VRAM | Rekomendasi |
| :---- | :---- | :---- | :---- | ----: | ----: | :---- |
| Scratch | CNN kecil | log-mel `4×64×T` | 128–256-d | ❌ | \<1 GB | ✅ baseline wajib |
| Scratch | CRNN | log-mel | 256-d | ❌ | 1–2 GB | ⚠️ opsional |
| Pretrained | YAMNet | waveform/log-mel | 1024-d | ✅ | \<2 GB | ✅ ringan |
| Pretrained | VGGish | log-mel | 128-d | ✅ | \<2 GB | ⚠️ legacy |
| Pretrained | OpenL3 | waveform/log-mel | 512-d | ✅ | \<2 GB | ✅ musik |
| Pretrained | PANNs CNN14 | log-mel | 2048-d | ✅ | 2–4 GB | ✅ kuat |
| Pretrained | AST | log-mel | 768-d | ✅ | 3–5 GB | ⚠️ opsional |
| Pretrained | MERT-330M | waveform | 1024-d | ✅ | 4–6 GB | ⚠️ frozen saja |
| Pretrained | CLAP | waveform/log-mel | 512-d | ✅ | 3–5 GB | ⚠️ audio–text, bukan gain |
| Pretrained | HuBERT/WavLM | waveform | 768-d | ✅ | 3–4 GB | ❌ speech bias |
| Pretrained | EnCodec | waveform | latent/token | ✅ | — | ❌ bukan regresi gain |

Rekomendasi audio:

```
Mulai : CNN scratch + log-mel
Naik  : frozen OpenL3 atau PANNs CNN14
Berat : MERT/CLAP/AST frozen, hanya jika waktu cukup
```

## 6.2 Text encoder

| Model | Dimensi | Berat | Bahasa | Mode | Rekomendasi |
| :---- | ----: | :---- | :---- | :---- | :---- |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Ringan | Multilingual | Frozen \+ MLP | ✅ paling praktis |
| `IndoBERT-base-p1` | 768 | Sedang | Indonesia | Frozen \+ MLP | ✅ alternatif utama |
| `multilingual-e5-small` | 384 | Ringan | Multilingual | Frozen \+ MLP | ✅ alternatif |
| `LaBSE` | 768 | Berat | Multilingual | Frozen only | ⚠️ berlebihan |
| `mBERT` | 768 | Berat | Multilingual | Frozen only | ❌ tidak perlu |
| `FastText` | 300 | Sangat ringan | Indonesia | Trainable | ⚠️ baseline lemah |

Rekomendasi teks:

```
Kecepatan + generalisasi : MiniLM multilingual
Akurasi Bahasa Indonesia : IndoBERT
```

## 6.3 Model utama

| Kode | Nama | Audio encoder | Text encoder | Fusion | Output | Peran | VRAM |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **M1** | Late Fusion Scratch | CNN scratch | MiniLM | Concat \+ MLP | `g*` (4) | Baseline | \<2 GB |
| **M2** | Late Fusion Pretrained | PANNs/OpenL3 frozen | MiniLM | Concat \+ MLP | `g*` (4) | Pretrained baseline | 2–4 GB |
| **M3** | Structured Two-Branch | CNN/PANNs | MiniLM/IndoBERT | `L` \+ `r` → rumus | `L`, `r`, `g*` | **Model utama** | 2–4 GB |
| **M4** | Cross-Attention | PANNs/OpenL3 frozen | MiniLM | Cross-attention | `g*` (4) | Upper-bound | 3–5 GB |
| **M5** | FiLM Conditioning | CNN scratch | MiniLM | FiLM | `g*` (4) | Bonus/opsional | \<2 GB |

Rekomendasi:

```
Minimal 3 model : M1 + M2 + M3
Rekomendasi 4   : M1 + M2 + M3 + M4
Bonus 5         : tambah M5
```

## 6.4 Detail M3 — Structured Two-Branch

### Arsitektur audio

```
log-mel vocals ─┐
log-mel drums  ─┤
log-mel bass   ─┼→ CNN_shared → L_pred (4)
log-mel other  ─┘
```

- Shared encoder per stem.  
- Parameter lebih sedikit.  
- Setiap stem diperlakukan sama.

Contoh CNN:

```
Conv2D 32 → Conv2D 64 → Conv2D 128
→ Global Average Pooling
→ FC 128
→ FC 1
```

### Arsitektur teks

```
prompt → frozen text encoder → MLP → r_pred (4)
```

### Fusion

```
L_ref  = mean(L_pred)
g_pred = g0 + (L_ref + r_pred - L_pred)
```

### Loss

```py
loss = MSE(g_pred, g_true)
     + 0.1 * MSE(L_pred, L_dsp)
     + 0.1 * MSE(r_pred, r_label)
```

Keterangan:

- `L_dsp` \= loudness dari RMS/LUFS.  
- `r_label` \= target balance dari prompt.  
- Auxiliary loss membantu audio branch belajar loudness dan text branch belajar prompt.

## 6.5 Detail M1

```
audio → CNN → audio_embedding (256) ─┐
                                     ├→ concat → MLP → g* (4)
prompt → text encoder → text_embedding (384) ┘
```

Kelebihan:

- sederhana,  
- baseline kuat,  
- mudah dilatih.

Kekurangan:

- fusion dangkal,  
- kurang interpretable.

## 6.6 Detail M2

Sama seperti M1, tetapi audio encoder memakai pretrained frozen:

```
log-mel → frozen PANNs/OpenL3 → embedding
```

Tujuan:

- mengukur apakah pretrained encoder membantu,  
- membandingkan scratch vs pretrained.

## 6.7 Detail M4

```
audio frames → keys/values
text tokens  → queries
   ↓
cross-attention
   ↓
pooled feature
   ↓
MLP → g* (4)
```

Kelebihan:

- ekspresif,  
- bisa menangkap hubungan kata dan bagian audio.

Kekurangan:

- lebih berat,  
- lebih mudah overfit,  
- butuh data lebih banyak.

## 6.8 Detail M5 — FiLM

```
prompt → text encoder → scale & shift
audio → CNN → feature map
feature map × scale + shift
   ↓
pooling → MLP → g* (4)
```

Kelebihan:

- conditioning ringan,  
- lebih baik dari concat biasa.

Kekurangan:

- lebih sulit dijelaskan,  
- butuh tuning.

---

# 7\. Training dan Inference

## 7.1 Training

Training dilakukan offline dari dataset:

```
(audio, prompt, g*)
```

Contoh batch:

```py
audio.shape  = (16, 4, 64, 800)
text.shape   = (16, 384)
target.shape = (16, 4)
```

Kode konseptual:

```py
for audio, prompt, g_true, L_dsp, r_label in dataloader:
    L_pred = audio_encoder(audio)
    r_pred = text_encoder(prompt)

    g_pred = g0 + (L_pred.mean(dim=1, keepdim=True) + r_pred - L_pred)

    loss = MSE(g_pred, g_true)
    loss += 0.1 * MSE(L_pred, L_dsp)
    loss += 0.1 * MSE(r_pred, r_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Hyperparameter awal:

| Parameter | Nilai |
| :---- | :---- |
| Batch size | 16 |
| Optimizer | AdamW |
| LR head | 1e-3 |
| LR encoder | 1e-4 jika fine-tune |
| Epochs | 30–50 |
| Early stopping | patience 5 |
| AMP | Ya |
| Gradient clipping | 1.0 |
| Seeds | 3 |

## 7.2 Inference

```py
L_pred = audio_encoder(audio)
r_pred = text_encoder(prompt)

g_pred = g0 + (L_pred.mean() + r_pred - L_pred)
g_pred = clamp(g_pred, -18, 12)

render_mix(stems, g_pred)
```

Di inference:

```
❌ tidak ada label
❌ tidak ada loss
❌ tidak ada backward
✅ hanya forward pass
```

## 7.3 Rendering

Model bekerja di 16 kHz mono, tetapi render final memakai stem asli:

```
stem asli 44.1 kHz stereo
   ↓
terapkan gain prediksi
   ↓
jumlahkan
   ↓
final_mix.wav
```

Cek clipping:

```
peak > 1.0 → clipping
```

Jika perlu:

- turunkan master gain,  
- atau gunakan limiter (catat sebagai mastering, bukan mixing).

## 7.4 Reproducibility

- Simpan seed.  
- Simpan config YAML.  
- Simpan versi library.  
- Simpan split manifest.  
- Simpan hasil evaluasi dalam CSV/JSON.  
- Jangan menimpa hasil eksperimen lama.

---

# 8\. Evaluasi

## 8.1 Metric level parameter

| Metric | Arti |
| :---- | :---- |
| MAE dB per stem | Rata-rata error gain |
| RMSE dB | Penalti error besar |
| Bias dB | Kecenderungan terlalu tinggi/rendah |
| R² | Variasi gain yang dijelaskan |
| Pearson/Spearman | Korelasi prediksi vs target |
| Max error dB | Worst-case |

## 8.2 Metric level spektral

| Metric | Arti |
| :---- | :---- |
| Band error 8-band | Selisih energi per band |
| Mean absolute band error | Error dalam dB |
| Log-spectral distance | Kemiripan spektrum |
| Spectral centroid difference | Keseimbangan terang/gelap |
| Worst-band error | Band paling sulit |

## 8.3 Metric level audio

| Metric | Alat | Target |
| :---- | :---- | :---- |
| Peak level | `soundfile` | ≤ 0 dBFS |
| Clipping rate | numpy | 0% |
| Integrated loudness | `pyloudnorm` | sesuai target |
| True peak | `ffmpeg loudnorm` | ≤ \-1 dBTP |
| Stereo correlation | numpy | tidak negatif ekstrem |
| Stereo width | side/mid ratio | sesuai |

## 8.4 Metric level teks

| Metric | Arti |
| :---- | :---- |
| Intent accuracy | Benar mengenali arah/aksi |
| Slot F1 | Benar mengenali stem dan magnitudo |
| Instruction adherence | Output sesuai prompt |
| Human rating | Penilaian 1–5 |
| Clarification rate | Berapa sering perlu bertanya |

## 8.5 Protocol evaluasi

- Split by song.  
- 3 seed.  
- Laporkan mean ± std / confidence interval.  
- Bandingkan M1, M2, M3, M4.  
- Ablation audio-only dan text-only.  
- Evaluasi pada 200 prompt manusia.  
- Uji per-song dan per-stem.  
- Jangan hanya melaporkan satu angka.

## 8.6 Success threshold

Jangan hanya memakai threshold lama `0.001`.

Rekomendasi:

```
success = MAE gain < 1.0 dB
```

atau:

```
success = band error < 0.01
```

Jelaskan threshold yang dipakai.

## 8.7 Baseline pembanding

- No-op (gain tidak diubah).  
- Random gain.  
- DSP/optimizer sederhana.  
- M1 scratch.  
- M2 pretrained.  
- M3 structured.  
- M4 cross-attention.

## 8.8 Human evaluation

Minimal:

```
5–10 listener
blind test
pertanyaan: mana yang lebih seimbang?
```

Tujuan:

- mengecek apakah metrik angka sesuai persepsi,  
- mengecek instruction adherence,  
- mengecek artefak/clipping.

---

# 9\. Deployment

## 9.1 Stack

```
Streamlit
PyTorch
torchaudio
transformers
sentence-transformers
soundfile
pyloudnorm
numpy
pandas
scikit-learn
matplotlib
seaborn
pyyaml
```

## 9.2 Alur aplikasi

```
Upload 4 stem
   ↓
Input prompt
   ↓
Preprocessing audio
   ↓
Model inference
   ↓
Prediksi 4 gain
   ↓
Render stereo mix
   ↓
Preview + download
```

## 9.3 Fitur minimum

- Upload vocals, drums, bass, other.  
- Input prompt.  
- Tombol `Mix`.  
- Tampilkan gain prediksi.  
- Preview audio.  
- Download `final_mix.wav`.  
- Tampilkan metrik sederhana.  
- Tampilkan error jika file tidak lengkap.

## 9.4 Menjalankan lokal

Contoh:

```sh
python3.12 -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 9.5 Catatan

- Deployment lokal sudah memenuhi syarat.  
- Cloud deployment bersifat bonus.  
- Jangan upload dataset berlisensi ke layanan publik tanpa cek izin.  
- Pastikan app tetap jalan tanpa GPU (fallback CPU) untuk demo.

---

# 10\. Pembagian Tugas 4 Orang

## 10.1 Hardware

| Anggota | CPU | RAM | GPU | Kekuatan | Batasan |
| :---- | :---- | :---- | ----: | :---- | :---- |
| **1** | Intel Core Ultra 5 125H | 16 GB | Intel Arc iGPU | CPU/OpenVINO, ringan | Tidak ada CUDA |
| **2** | Ryzen 5 7535HS | 24 GB | RTX 3050 **4 GB** | CUDA kecil, RAM cukup | VRAM paling sempit |
| **3** | i7-14700HX | 32 GB | RTX 4050 **6 GB** | CPU/RAM paling kuat | VRAM sedang |
| **4 (kamu)** | i5-13450HX | 16 GB | RTX 5050 **8 GB** | VRAM paling besar | RAM paling kecil |

## 10.2 Hasil cek spek anggota 4

| Komponen | Spek |
| :---- | :---- |
| CPU | Intel Core i5-13450HX, 10 core / 16 thread |
| RAM | 15 GiB (16 GB), swap 30 GiB |
| GPU | NVIDIA RTX 5050 Laptop, 8 GB VRAM (8151 MiB) |
| Driver/CUDA | Driver 615.78.08, CUDA 13.4 |
| Disk | `/` 172 GB free, `/mnt/shared` 26 GB free |

Kesimpulan:

```
VRAM paling besar di kelompok
RAM paling kecil
Cocok untuk training multimodal berat dan deployment GPU
Jangan load seluruh dataset ke RAM
Cache di /home/ilydien atau /
```

## 10.3 Role

### Anggota 1 — Data, Teks, dan Dokumentasi

**Peran:** Data & Text Pipeline Lead

Tugas:

1. Download/validasi MUSDB18-HQ.  
2. Buat split manifest 80/20/50.  
3. Buat pipeline preprocessing.  
4. Buat prompt generator dan mapping prompt → `r`.  
5. Buat 200 prompt human-written untuk test.  
6. Buat README, requirements, dokumentasi deploy.  
7. Bantu Bab I dan Bab III bagian dataset.

Output:

```
data/splits/musdb18hq_v1.json
data/prompts/train.json
data/prompts/test_human.json
README.md
requirements.txt
```

### Anggota 2 — Baseline dan Ablation

**Peran:** Baseline & Ablation Lead

Tugas:

1. Implementasi M1.  
2. Training baseline batch 4–8, AMP, gradient accumulation.  
3. Ablation audio-only dan text-only.  
4. Hyperparameter tuning kecil.  
5. Tabel hasil baseline.  
6. Bantu Bab II dan Bab IV bagian baseline.

Output:

```
models/m1_late_fusion.py
configs/m1.yaml
outputs/m1_results.csv
```

Catatan: 4 GB VRAM, jangan pakai MERT/CLAP/AST.

### Anggota 3 — Model Utama dan Evaluasi

**Peran:** Main Model & Evaluation Lead

Tugas:

1. Implementasi M3.  
2. Implementasi M2.  
3. Training dan tuning batch 8–16.  
4. Evaluasi MAE/RMSE, band error, success, clipping.  
5. Error analysis per stem, prompt, dan lagu.  
6. Bab III metodologi dan Bab IV hasil utama.

Output:

```
models/m3_structured.py
models/m2_pretrained_fusion.py
outputs/main_results.csv
outputs/error_analysis.md
```

Catatan: 32 GB RAM berguna untuk DataLoader dan cache.

### Anggota 4 — Multimodal Lanjutan dan Deployment

**Peran:** Multimodal & Deployment Lead

Tugas:

1. Implementasi M4 cross-attention.  
2. Eksperimen CLAP/MERT/AST frozen.  
3. Optimasi inference (AMP, quantization jika perlu).  
4. Bangun Streamlit.  
5. Integrasi semua model ke repo.  
6. Bab IV deployment dan Bab V.  
7. Koordinasi teknis: review PR, config, reproducibility.

Output:

```
models/m4_cross_attention.py
app/streamlit_app.py
outputs/deployment/
docs/how_to_deploy.md
```

Catatan: 8 GB VRAM, tetapi RAM 16 GB. Batasi worker dan gunakan cache disk.

## 10.4 Model ownership

| Model | Pemilik | Backup |
| :---- | :---- | :---- |
| M1 Late Fusion Scratch | Anggota 2 | Anggota 1 |
| M2 Pretrained Fusion | Anggota 3 | Anggota 4 |
| M3 Structured Two-Branch | Anggota 3 | Anggota 4 |
| M4 Cross-Attention | Anggota 4 | Anggota 3 |
| Data & Prompt | Anggota 1 | Anggota 2 |
| Evaluasi | Anggota 3 | Anggota 2 |
| Deployment | Anggota 4 | Anggota 1 |

## 10.5 Pembagian laporan

| Bab | PIC | Dibantu |
| :---- | :---- | :---- |
| Bab I Pendahuluan | Anggota 1 | Semua |
| Bab II Tinjauan Pustaka | Anggota 2 | Anggota 3 |
| Bab III Metodologi | Anggota 3 | Anggota 1 |
| Bab IV Hasil | Anggota 3 | Anggota 2, 4 |
| Bab IV Deployment | Anggota 4 | Anggota 1 |
| Bab V Kesimpulan | Anggota 4 | Semua |
| Daftar Pustaka | Anggota 1 | Semua |
| Formatting IEEE | Anggota 1 | Semua |

## 10.6 Git workflow

Branch:

```
main
├── feat/data
├── feat/m1-baseline
├── feat/m3-structured
├── feat/m4-cross-attention
└── feat/deployment
```

Aturan:

- Jangan commit dataset besar.  
- Simpan hasil di `outputs/`.  
- Konfigurasi di `configs/*.yaml`.  
- Setiap eksperimen catat model, seed, split, metrik, waktu.  
- Minimal 1 reviewer per PR.  
- Jangan menimpa hasil lama.

---

# 11\. Timeline

## 11.1 Timeline 4 minggu

| Minggu | Anggota 1 | Anggota 2 | Anggota 3 | Anggota 4 |
| :---- | :---- | :---- | :---- | :---- |
| **1** | Dataset \+ split \+ prompt | M1 skeleton | M3 skeleton | Deployment skeleton |
| **2** | Prompt test \+ dokumentasi | Training M1 | Training M2/M3 | M4 \+ CLAP test |
| **3** | Validasi data | Ablation | Evaluasi utama | Integrasi \+ app |
| **4** | Laporan Bab I/III | Laporan Bab II | Laporan Bab III/IV | Laporan IV/V \+ demo |

## 11.2 Milestone

| Milestone | Target |
| :---- | :---- |
| M1: Dataset siap | Akhir minggu 1 |
| M2: Baseline jalan | Akhir minggu 1 |
| M3: Model utama jalan | Akhir minggu 2 |
| M4: Evaluasi awal | Akhir minggu 2 |
| M5: Deployment lokal | Akhir minggu 3 |
| M6: Laporan lengkap | Akhir minggu 4 |
| M7: Demo | Akhir minggu 4 |

## 11.3 Buffer

Sisakan 2–3 hari buffer untuk:

- bug training,  
- OOM,  
- revisi laporan,  
- deployment.

---

# 12\. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
| :---- | :---- | :---- |
| Prompt membocorkan `g*` | Text-only menang | Prompt → `r`, bukan `g*` |
| Target bands di input | Label leakage | Jangan masukkan target bands |
| Split by excerpt | Data leakage | Split by song |
| Template sama di train/test | Menghafal template | Split by template \+ human test |
| Dataset kecil | Overfit | Frozen encoder, dropout, early stopping |
| 4 GB VRAM | OOM | Batch kecil, AMP, gradient accumulation |
| 8 GB VRAM | OOM pada model besar | Frozen encoder, quantization |
| RAM 16 GB | DataLoader penuh | Cache disk, worker kecil |
| `/mnt/shared` penuh | Gagal simpan | Cache di `/home/ilydien` |
| Label sintetis | Tidak mencerminkan manusia | Listening test \+ batasan jelas |
| MusikDB license | Tidak boleh komersial | Jelaskan batasan riset |
| Scope melebar | Tidak selesai | Kunci gain-only |
| Success rate rendah | Metrik tidak informatif | Laporkan MAE/RMSE |
| Perbandingan tidak adil | Kritik penguji | Semua model supervised, split sama |

---

# 13\. Mapping Laporan

## Bab I — Pendahuluan

- Latar belakang: mixing butuh keahlian, instruksi teks lebih aksesibel.  
- Rumusan masalah.  
- Tujuan.  
- Batasan.  
- Manfaat SDG 4 dan 17\.

## Bab II — Tinjauan Pustaka

- Machine learning dan deep learning.  
- CNN, CRNN, Transformer.  
- Text encoder dan multimodal fusion.  
- Log-mel, loudness, gain staging.  
- Metrik evaluasi.  
- Minimal 5 penelitian 5 tahun terakhir.  
- Perbedaan dengan penelitian sebelumnya.  
- Kontribusi.

## Bab III — Metodologi

- Diagram alur penelitian.  
- Dataset MUSDB18-HQ.  
- Preprocessing.  
- Pembuatan label multimodal.  
- Model M1–M4.  
- Hyperparameter tuning.  
- Evaluasi.

## Bab IV — Hasil dan Pembahasan

- Hasil eksperimen.  
- Perbandingan model.  
- Analisis error.  
- Deployment.  
- Screenshot aplikasi.

## Bab V — Kesimpulan dan Saran

- Jawaban rumusan masalah.  
- Apakah tujuan tercapai.  
- Rekomendasi pengembangan.

## Referensi

- Minimal 10 paper 5 tahun terakhir.  
- Minimal 2 buku.  
- Format IEEE.  
- Ikuti format font/spasi kampus.

---

# 14\. Checklist

## Data

- [ ] MUSDB18-HQ tervalidasi.  
- [ ] Split by song dibuat.  
- [ ] Preprocessing 16 kHz mono.  
- [ ] Log-mel cache dibuat.  
- [ ] Prompt train dibuat.  
- [ ] 200 prompt human test dibuat.  
- [ ] Schema dataset konsisten.

## Model

- [ ] M1 baseline.  
- [ ] M2 pretrained.  
- [ ] M3 structured.  
- [ ] M4 cross-attention.  
- [ ] M5 FiLM (opsional).  
- [ ] Ablation audio-only.  
- [ ] Ablation text-only.  
- [ ] 3 seed.

## Evaluasi

- [ ] MAE/RMSE per stem.  
- [ ] Band error.  
- [ ] Success threshold jelas.  
- [ ] Clipping rate.  
- [ ] LUFS/true peak.  
- [ ] Per-song.  
- [ ] Human test.  
- [ ] Error analysis.

## Deployment

- [ ] Streamlit jalan.  
- [ ] Upload 4 stem.  
- [ ] Input prompt.  
- [ ] Render mix.  
- [ ] Download.  
- [ ] README.  
- [ ] requirements.txt.  
- [ ] Fallback CPU.

## Laporan

- [ ] Bab I–V.  
- [ ] IEEE.  
- [ ] 10 paper \+ 2 buku.  
- [ ] Daftar tabel/gambar.  
- [ ] Screenshot deployment.  
- [ ] Notebook runnable.

---

# 15\. Lampiran

## 15.1 Glosarium

| Istilah | Arti |
| :---- | :---- |
| **Stem** | Track terpisah (vocals, drums, bass, other) |
| **Mixture** | Hasil penjumlahan stem |
| **Gain** | Penguatan sinyal dalam dB |
| **Loudness** | Tingkat keras yang terukur/terdengar |
| **dBFS** | Skala digital relatif terhadap full scale |
| **LUFS** | Loudness perseptual |
| **RMS** | Energi rata-rata sinyal |
| **r** | Target keseimbangan dari prompt |
| **L** | Loudness stem saat ini |
| **g**\* | Gain target |
| **L\_ref** | Rata-rata loudness |
| **Leakage** | Informasi target masuk ke input |
| **Stem mixing** | Menyeimbangkan stem sebelum dijumlahkan |

## 15.2 Rumus ringkas

```
L_ref = mean(L)

g*_i = g0_i + (L_ref + r_i - L_i)

sum(r_i) = 0

g* = clip(g*, -18, 12)
```

## 15.3 Shape ringkas

| Data | Shape |
| :---- | :---- |
| Audio per stem | `(1, 64, 800)` |
| Audio 4 stem | `(4, 64, 800)` |
| Text embedding | `(384,)` atau `(768,)` |
| `L` | `(4,)` |
| `r` | `(4,)` |
| `g*` | `(4,)` |
| Batch audio | `(B, 4, 64, 800)` |
| Batch text | `(B, 384)` |
| Batch target | `(B, 4)` |

## 15.4 Struktur folder yang diusulkan

```
wadidaw-studio/
├── data/
│   ├── raw/
│   │   └── musdb18hq/
│   ├── splits/
│   │   └── musdb18hq_v1.json
│   ├── prompts/
│   │   ├── train.json
│   │   └── test_human.json
│   └── cache/
│       └── logmel/
├── wadidaw/
│   ├── data/
│   │   ├── musdb.py
│   │   ├── prompts.py
│   │   └── features.py
│   ├── models/
│   │   ├── audio_cnn.py
│   │   ├── text_encoder.py
│   │   ├── fusion.py
│   │   └── structured.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── app/
│       └── streamlit_app.py
├── configs/
│   ├── m1.yaml
│   ├── m2.yaml
│   ├── m3.yaml
│   └── m4.yaml
├── outputs/
│   ├── m1_results.csv
│   ├── main_results.csv
│   └── deployment/
├── docs/
│   └── how_to_deploy.md
├── requirements.txt
├── README.md
└── dokumentasi.md
```

## 15.5 Perintah awal

```sh
python3.12 -m venv .venv
source .venv/bin/activate.fish
pip install --upgrade pip
pip install -r requirements.txt
```

Cek GPU:

```sh
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Menjalankan app:

```sh
streamlit run app/streamlit_app.py
```

## 15.6 Catatan akhir

Dokumen ini adalah **ringkasan keputusan desain**. Angka dan konfigurasi dapat berubah setelah eksperimen. Yang harus dijaga:

1. Tidak ada label leakage.  
2. Tidak ada data leakage.  
3. Model utama memakai audio dan teks.  
4. Perbandingan model adil.  
5. Deployment lokal jalan.  
6. Laporan jujur soal keterbatasan.