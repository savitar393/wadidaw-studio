# Menjalankan eksperimen audio encoder di Kaggle

Folder ini berisi **versi Kaggle** dari `notebooks/1_audio_encoder_selection.ipynb`.

- `1_audio_encoder_selection_kaggle.ipynb` — notebook siap import ke Kaggle.
- Logika eksperimen (M3 structured head, loss, seed, effective batch 16, metrik) **identik** dengan
  notebook lokal. Yang ditambahkan hanya bootstrap repo, deteksi dataset/cache, resume, dan budget guard.

Satu notebook dipakai untuk **semua member** (1–4). Kamu hanya mengganti `MEMBER` di Cell 1
(atau set environment variable `WADIDAW_MEMBER`).

---

## 1. Prasyarat (sekali saja)

### 1.1 Dataset audio

Dataset yang dipakai:
[`satriamanggala/musdb18hq`](https://www.kaggle.com/datasets/satriamanggala/musdb18hq) (±31,2 GB).

Di halaman notebook Kaggle:

1. Panel kanan → **Add Input** → **Datasets**.
2. Cari `satriamanggala/musdb18hq` → **Add**.

Notebook mendeteksi otomatis folder yang punya `train/` + `test/` + `*/mixture.wav`,
termasuk kalau strukturnya bersarang (`.../MUSDB18-HQ/train/...`) atau masih berupa `.zip`
(zip akan diekstrak ke `/kaggle/tmp`).

### 1.2 Cache log-mel (disarankan, ±593 MB)

Cache ini **tidak ikut `git clone`** karena `data/cache/` di-`.gitignore`.
Upload sebagai **private dataset** supaya tiap sesi tidak perlu membangun ulang 30–60 menit.

Di lokal:

```bash
# struktur yang diharapkan di dalam dataset:
# data/cache/logmel/train/*.npy
# data/cache/logmel/validation/*.npy
# data/cache/logmel/test/*.npy
```

Lalu di Kaggle:

1. **Datasets → New Dataset**.
2. Nama, misal: `wadidaw-logmel-cache`.
3. Drag folder `data/cache/logmel` (folder `logmel` itu sendiri) ke area upload.
4. **Create**.
5. Di notebook: **Add Input** → pilih dataset tersebut.

Notebook menerima dua struktur:

```
wadidaw-logmel-cache/logmel/train/*.npy     # folder logmel ikut ter-upload
wadidaw-logmel-cache/train/*.npy            # isi logmel langsung di root dataset
```

Kalau cache tidak ditemukan dan `BUILD_LOGMEL_IF_MISSING = True`, notebook membangunnya
dari WAV (butuh 30–60 menit, hanya untuk sesi itu).

Cache waveform 16 kHz (±1,5 GB) **tidak perlu di-upload** — dibangun otomatis oleh Cell 2
(hanya membaca excerpt 8 detik, bukan seluruh lagu).

### 1.3 Setting sesi

| Opsi | Nilai |
| --- | --- |
| Accelerator | `GPU T4 x2` (atau `P100`) |
| Internet | **ON** (wajib: `git clone`, IndoBERT, AST/MERT, BEATs) |
| Persistence | default |

---

## 2. Memasukkan notebook ke Kaggle

### Cara web (dipakai di sini)

1. Buka [kaggle.com](https://www.kaggle.com) → **Create → Notebook**.
2. **File → Import Notebook** → upload `1_audio_encoder_selection_kaggle.ipynb`.
3. **Add Input** → dataset audio + dataset cache log-mel.
4. **Session options** → GPU T4 x2, Internet ON.

### Alternatif CLI (opsional)

```bash
# butuh ~/.kaggle/kaggle.json
kaggle kernels init -p notebooks/kaggle
kaggle kernels push -p notebooks/kaggle
```

---

## 3. Menjalankan per sesi

Urutan sel:

1. **Setup** — clone repo, install dependensi, deteksi dataset, symlink, verifikasi 150 lagu.
2. **Cache** — attach cache log-mel, verifikasi semua `audio_path` manifest ada.
3. **Cell 1** — konfigurasi + `MEMBER`.
4. **Cell 2–8** — cache waveform, dataset, fitur level, text branch, encoder, model, train loop.
5. **Cell 9** — latih shard (`RUN_EXPERIMENT = True`).
6. **Cell 10–12** — leaderboard, figure, parity checks.
7. **Cell 13** — zip hasil ke `/kaggle/working/encoder_selection_member<N>.zip`.

### Konfigurasi penting di Cell 1

```python
MEMBER = 1                     # 1/2/3/4
RUN_PLAN = None                # None = semua run shard; atau [("ast", "bimodal", 42)]
MAX_RUNS_PER_SESSION = 0       # 0 = tanpa batas; 1–2 untuk AST/MERT/BEATs
TIME_BUDGET_MIN = 660          # stop rapi sebelum sesi 12 jam
```

### Skenario yang disarankan

| Sesi | `MEMBER` | `MAX_RUNS_PER_SESSION` | Isi |
| --- | --- | --- | --- |
| 1 | 1 | 0 | `cnn_scratch` × 2 mode × 3 seed (6 run, paling ringan — sekaligus smoke test) |
| 2–4 | 2 | 2 | `ast` + `text_only` (9 run) |
| 5–7 | 3 | 2 | `mert` (6 run) |
| 8–10 | 4 | 2 | `beats` (6 run) |

Estimasi kasar di T4: `cnn_scratch` 15–30 menit/run; `ast`/`mert`/`beats` 2,5–4 jam/run;
`text_only` jauh lebih cepat. Kuota GPU Kaggle ±30 jam/minggu, jadi jangan paksa satu sesi.

### Resume

Run yang JSON-nya sudah ada di `outputs/encoder_selection/member<N>/` otomatis dilewati.
Kalau sesi berhenti di tengah, buka lagi notebook, jalankan Setup → Cache → Cell 1–9,
dan training melanjutkan run yang belum selesai.

Agar progres tidak hilang, **download zip Cell 13 / Save Version** setiap akhir sesi,
lalu commit JSON-nya ke repo:

```bash
outputs/encoder_selection/member<N>/<encoder>_<mode>_seed<seed>.json
```

Ketua kemudian `git pull` dan menjalankan Cell 10 untuk menggabungkan semua member.

---

## 4. Troubleshooting

| Gejala | Solusi |
| --- | --- |
| `MUSDB18-HQ tidak ditemukan` | Pastikan dataset sudah **Add Input**. Kalau strukturnya tak biasa, cek output `contoh folder dataset` lalu sesuaikan `find_musdb_root`. |
| `cache log-mel tidak lengkap` | Upload ulang `data/cache/logmel` atau set `BUILD_LOGMEL_IF_MISSING = True`. |
| `Repository not found` saat clone | Repo privat? Ganti `REPO_URL` ke URL dengan token, atau upload repo sebagai dataset. |
| MERT gagal `trust_remote_code` | Pin `transformers` yang kompatibel di sel Setup (mis. `pip install -q transformers==4.40.2`). |
| BEATs gagal di-install | Butuh Internet ON dan `git`; cek pesan error dari `pip`. |
| CUDA OOM pada AST/BEATs | Turunkan `batch_size` dan naikkan `grad_accum_steps` agar hasil kali tetap 16 (mis. 2×8), lalu catat di laporan. |
| Sesi mati sebelum selesai | Turunkan `MAX_RUNS_PER_SESSION` / `TIME_BUDGET_MIN`; gunakan resume. |
| Output >20 GB | Notebook ini hanya menulis JSON kecil. Cache (±2,1 GB) ada di `/kaggle/working` tapi tidak di-zip; input 31 GB tidak dihitung ke kuota working. |

---

## 5. Catatan fairness

Yang berubah antar member hanya blok audio encoder. Text encoder (IndoBERT di-fine-tune),
head M3, loss, optimizer, seed, effective batch 16, dan metrik tetap sama.
Jangan mengubah `CONFIG` selain kebutuhan OOM yang didokumentasikan (aturan `batch_size × grad_accum_steps = 16`).
