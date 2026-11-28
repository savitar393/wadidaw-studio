Desain ini **sudah sangat kuat dan rapi secara metodologi**. Pemisahan antara baseline untuned, tuning unimodal (Stage 1 dan 2), tuning bimodal (Stage 3 dan 4 dengan Top-2), serta evaluasi akhir 3 seed di test set sudah menjawab hampir semua celah metodologis yang sebelumnya ada.

Ada **3 koreksi/klarifikasi penting** yang perlu dirapikan supaya tidak menimbulkan pertanyaan kritis dari reviewer atau penguji, plus 1 trik efisiensi yang menghemat banyak waktu komputasi.

---

### 1. Koreksi di Ablation Akhir Test: Judul vs Isi Tertukar (dan Missing Baseline)

Di poin terakhir **Ablation Akhir Test**, kamu menulis:
> **Judul:** `Bimodal (train biasa, default parameter)`
> **Isi:** `1 Bimodal terdiri dari Audio dengan hyperparameter terbaik dari Stage 3 dan Text dengan hyperparameter terbaik dari Stage ini`

Judul dan isinya bertentangan. Di **Ablation Akhir Test**, kamu justru **sebaiknya menyertakan keduanya** sebagai dua baris terpisah (masing-masing 3 seed):
1. **Bimodal Baseline (default parameter, encoder pemenang):** untuk membuktikan bahwa tuning OFAT di Stage 3 + Stage 4 benar-benar memberi peningkatan di data test (bukan sekadar overfit di validation).
2. **Bimodal Final Tuned (Audio Stage 3 + Text Stage 4):** sebagai model utama yang diusulkan.

---

### 2. Perjelas Makna "Hasil Bimodal, tapi Audio / Text-nya Dihapus"

Pada poin 3 dan 4 di Ablation Akhir Test, frasa *"hasil Bimodal, tapi Audio/Text-nya dihapus"* bisa ditafsirkan dengan **dua cara berbeda**:

- **Tafsiran A (Retrain dari nol di mode unimodal memakai hyperparameter Stage 3 / Stage 4):**
  Ini **kurang disarankan** sebagai ablasi utama. Kamu sudah punya Stage 1 (`audio_only` tuned optimal) dan Stage 2 (`text_only` tuned optimal). Melatih ulang `audio_only` memakai hyperparameter milik `bimodal` (Stage 3) kurang adil karena hyperparameter tersebut dioptimalkan untuk dinamika gradien bimodal, bukan unimodal. Ini hanya berguna kalau pertanyaanmu adalah *cross-mode transfer hyperparameter* (dan memakan 6 run training tambahan).

- **Tafsiran B (Inference-time modality removal pada checkpoint Bimodal Final, TANPA retrain) — Direkomendasikan:**
  Ambil checkpoint **Bimodal Final (Stage 4, 3 seed)** yang sudah jadi, lalu saat evaluasi di test set matikan salah satu cabangnya (`l_pred = 0` untuk menghapus audio, atau `r_pred = 0` untuk menghapus teks).
  - **Biaya komputasi: 0 training run** (hanya inferensi beberapa detik).
  - **Pertanyaan yang dijawab berbeda dan sangat tajam:**
    - *Stage 1 & Stage 2 (retrained unimodal)* menjawab: **"Apakah bimodal mengalahkan model unimodal terbaik yang dilatih khusus?"**
    - *Inference-time removal pada Bimodal Final* menjawab: **"Apakah model bimodal yang sudah dilatih benar-benar memanfaatkan kedua modalitas, atau diam-diam hanya mengandalkan satu cabang saja?"**
  - Di arsitektur M3 (`g = L_ref + r - L`), interpretasinya sangat jelas: mematikan teks menghasilkan *auto-leveling* rata berbasis audio (`L_ref - L`), sedangkan mematikan audio menghasilkan *target balance* murni dari teks (`r`) tanpa mengetahui loudness awal stem.

---

### 3. Dua Catatan Kecil yang Perlu Dikunci di Protokol

1. **Bagaimana jika pemenang Stage 1 (`audio_only`) berbeda dengan pemenang Stage 4 (`bimodal`)?**
   - Misal di `audio_only` pemenangnya `MERT`, tetapi di `bimodal` pemenangnya `BEATs`.
   - Aturan yang paling defensible:
     - Jika sama: cukup 1 model `Unimodal Audio` (3 seed).
     - Jika berbeda: laporkan `Unimodal Audio` terbaik Stage 1 (*best unimodal vs best bimodal*), **dan** tambahkan 1 baris `Unimodal Audio` (hasil Stage 1) untuk encoder pemenang bimodal agar perbandingan ablasi pada **encoder yang sama** tetap ada.
2. **Redaksi Stage 4:**
   Di judul tertulis `[sekali, text hanya satu jenis]`, sedangkan isinya menjalankan OFAT teks pada **Top-2 encoder** dari Stage 3. Ubah catatannya menjadi: *"Text hanya satu jenis (IndoBERT), di-OFAT pada masing-masing Top-2 encoder dari Stage 3 (paralel di 2 anggota)"* supaya tidak terkesan kontradiktif.
3. **Pastikan "Ablation Awal & Baseline" memakai Validation Set saja:**
   Jangan jalankan evaluasi test set pada tahap awal agar test set tetap 100% steril sampai **Ablation Akhir Test**.

---

### 4. Trik Efisiensi Gratis (Deduplikasi Baseline)

Karena semua titik di **Ablation Awal & Baseline** menggunakan parameter default `CONFIG` (dan seed tuning `42`), dan nilai default tersebut termasuk dalam ruang pencarian OFAT:
- Hasil baseline dari tahap awal bisa **langsung dipakai ulang (*reuse / alias*)** sebagai titik awal di **Stage 1, Stage 2, dan Stage 3** tanpa melatih ulang.
- Begitu juga di **Stage 4**: titik awal teks default sudah dihitung sebagai pemenang Stage 3, jadi tidak perlu dijalankan ulang.
- Ini **menghemat 11 run training penuh** tanpa mengubah hasil sedikit pun.

---

### Ringkasan Desain Final yang Sudah Disempurnakan

Dengan asumsi 3 knob per cabang (Audio: `3×2×3` = 6 kombinasi unik; Text: `3×3×3` = 7 kombinasi unik) dan deduplikasi titik baseline:

| Fase | Eksperimen | Data & Seed | Run Unik Baru | Eksekusi |
|---|---|---|---|---|
| **Baseline Awal** | `text_only` (1), `audio_only` (4), `bimodal` (4) default | Val, 1 seed (`42`) | 9 run (2–3/anggota) | Paralel 4 anggota |
| **Stage 1** | Audio Unimodal OFAT (4 encoder) | Val, 1 seed (`42`) | +5 run / encoder (20 run) | Paralel 4 anggota |
| **Stage 2** | Text Unimodal OFAT (1x) | Val, 1 seed (`42`) | +6 run (1 anggota) | Paralel di Phase 1 |
| **Stage 3** | Bimodal Audio OFAT, Text default (4 encoder) | Val, 1 seed (`42`) | +5 run / encoder (20 run) | Paralel 4 anggota |
| **Stage 4** | Bimodal Text OFAT pada **Top-2** Stage 3 | Val, 1 seed (`42`) | +6 run / Top-2 (12 run) | Paralel 2 anggota |
| **Ablation Akhir Test** | 1. `Unimodal Audio` (best Stage 1)<br>2. `Unimodal Text` (best Stage 2)<br>3. `Bimodal Baseline` (default, encoder pemenang)<br>4. `Bimodal Final` (Audio Stage 3 + Text Stage 4) | **Test** (`human` + `synthetic`), **3 seed** (`42, 123, 2026`) | 4 × 3 = 12 run (3/anggota) | Paralel 4 anggota |
| **Modality Reliance Test** | Checkpoint `Bimodal Final` (3 seed) dengan cabang audio / teks dimatikan saat inferensi | **Test**, 3 seed | **0 run training** (inferensi saja) | 1 anggota (beberapa detik) |

**Kesimpulan:** Setelah memisahkan `Bimodal Baseline` vs `Bimodal Final` di test set dan menjadikan "hapus audio/teks dari Bimodal" sebagai *inference-time modality removal*, desain ini sudah **sangat kokoh untuk level skripsi, laporan proyek, maupun paper**: bebas leakage, adil antar-encoder, adil antar-modalitas, menguji efek tuning, menguji ketergantungan modalitas, dan tetap realistis (sekitar 15 sampai 26 run per anggota).


# Penjelasan Tafsiran B (Inference Time modality removal)

Intinya sederhana: **kamu tidak melatih model baru**. Kamu hanya mengambil **file checkpoint model Bimodal Final yang sudah jadi** (hasil Stage 4), lalu saat evaluasi di data test, salah satu cabangnya "dimatikan" untuk melihat seberapa besar error-nya naik.

### Analogi Sederhana
Bayangkan seorang *sound engineer* (model Bimodal Final yang sudah selesai belajar) yang biasanya nge-mix dengan **mendengar audio** dan **membaca instruksi teks klien**:
- **Normal (Bimodal):** Dengar audio + baca teks. Hasilnya paling akurat (misal error kecil).
- **Audio dihapus saat tes:** Orang yang **sama** (tanpa dilatih ulang) diminta nge-mix dengan telinga ditutup, hanya membaca teks.
- **Teks dihapus saat tes:** Orang yang **sama** diminta nge-mix tanpa membaca instruksi klien, hanya mendengar audio.

**Apa yang dibuktikan dari tes ini?**
- Kalau saat audio dimatikan error melonjak drastis, **dan** saat teks dimatikan error juga melonjak drastis, itu **bukti nyata** bahwa model bimodalmu benar-benar memanfaatkan kedua modalitas secara aktif.
- Sebaliknya, kalau saat teks dimatikan error-nya ternyata hampir tidak berubah, artinya selama training model bimodalmu diam-diam mengabaikan teks dan hanya mengandalkan audio (fenomena *modality collapse*).

---

### Cara Kerjanya di Kode M3 Proyek Ini
Pada `EncoderProbeModel` di [1_audio_encoder_selection_colab2.ipynb](file:///home/acer/dev/wadidaw-studio/notebooks/colab/1_audio_encoder_selection_colab2.ipynb), prediksi gain dihitung dengan rumus:

$$g_{\text{pred}} = L_{\text{ref}} + r_{\text{pred}} - L_{\text{pred}}$$

- $L_{\text{pred}}$ (dan rata-ratanya $L_{\text{ref}}$) keluar dari **cabang audio**.
- $r_{\text{pred}}$ keluar dari **cabang teks**.

Saat mengevaluasi checkpoint **Bimodal Final** di test set, kamu tinggal menjalankan fungsi evaluasi 3 kali pada checkpoint yang sama dengan mengganti `model.mode` sementara (tanpa training ulang):

1. `model.mode = "bimodal"` (normal): $g_{\text{pred}} = L_{\text{ref}} + r_{\text{pred}} - L_{\text{pred}}$
2. `model.mode = "text_only"` (cabang audio dimatikan, $L_{\text{pred}} = 0$): $g_{\text{pred}} = r_{\text{pred}}$. Model hanya menebak *balance* dari teks tanpa tahu seberapa keras/pelan stem aslinya.
3. `model.mode = "audio_only"` (cabang teks dimatikan, $r_{\text{pred}} = 0$): $g_{\text{pred}} = L_{\text{ref}} - L_{\text{pred}}$. Model hanya meratakan volume semua stem (*auto-leveling*) tanpa tahu instruksi prompt.

Karena tidak ada training ulang, proses ini hanya butuh waktu inferensi beberapa detik saja.

---

### Bedanya dengan Unimodal Stage 1 & Stage 2

| Metode | Prosesnya | Pertanyaan yang Dijawab |
|---|---|---|
| **Retrained Unimodal** (Stage 1 & Stage 2) | Melatih model **baru dari nol** khusus 1 modalitas dengan hyperparameter terbaiknya sendiri | *"Apakah sistem bimodal mengalahkan sistem unimodal terbaik?"* |
| **Inference-time Removal** (Tafsiran B) | Memakai **checkpoint Bimodal Final yang sama**, lalu mematikan 1 cabang saat evaluasi | *"Apakah di dalam model bimodal itu sendiri, cabang audio dan cabang teks benar-benar sama-sama dipakai?"* |

Jadi kalau kamu menampilkan keduanya di **Ablation Akhir Test**, analisisnya jadi lengkap tanpa membuang biaya training tambahan untuk melatih ulang unimodal memakai hyperparameter milik bimodal.