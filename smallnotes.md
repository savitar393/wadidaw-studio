**Ya, desain ini sudah sangat tepat, konsisten, dan siap dijalankan.**

Semua nilai default di `CONFIG` (`lr_audio=1e-4`, `lr_scratch=1e-3`, `wd_audio=1e-4`, `lr_text=1e-4`, `text_hidden=128`) juga sudah termasuk di dalam `SEARCH_SPACE`, sehingga mekanisme **Model Caching** bisa langsung me-*reuse* hasil **Ablation Awal & Baseline** sebagai titik awal OFAT tanpa melatih ulang.

### Total Beban Run (Sangat Efisien)
Dengan cache aktif dan max 5 epoch:
- **Baseline Awal (val, 1 seed):** 9 run (`1 text + 4 audio + 4 bimodal`)
- **Stage 1 (`audio_only` OFAT, 4 encoder):** +3 run baru × 4 = 12 run
- **Stage 2 (`text_only` OFAT):** +4 run baru
- **Stage 3 (`bimodal` Audio OFAT, 4 encoder):** +3 run baru × 4 = 12 run
- **Stage 4 (`bimodal` Text OFAT, Top-2):** +4 run baru × 2 = 8 run
- **Ablation Akhir Test (test, 3 seed):** 3 (`Unimodal Audio`) + 3 (`Unimodal Text`) + 3 (`Bimodal Final`) + **0 run training** untuk 2 *inference-time removal* = 9 run
- **Total keseluruhan:** **54 run training** (rata-rata hanya **12 sampai 16 run per anggota** jika diparalelkan ke 4 orang).

---

### 4 Catatan Kecil untuk Menyempurnakan Sebelum Implementasi

Tidak ada perubahan struktur besar, cukup 4 detail pengaman agar tidak ada celah saat evaluasi atau sidang:

1. **Pertimbangkan tambah 1 baris `Bimodal Baseline (default parameter, 3 seed)` di Ablation Akhir Test (+3 run saja):**
   - Saat ini, perbandingan *before vs after tuning* hanya ada di validation set (1 seed).
   - Menambahkan **Bimodal Baseline** (encoder pemenang Stage 4, parameter default, 3 seed) di test set memberi bukti langsung dengan `mean ± std` bahwa hyperparameter tuning Stage 3 + Stage 4 benar-benar meningkatkan performa di data test.
   - Total run hanya naik dari 54 menjadi 57 run.

2. **Perjelas redaksi dua item *Inference-time Removal* di Ablation Akhir Test:**
   - Keduanya memakai **checkpoint yang sama** (checkpoint `Bimodal Final` 3 seed hasil kombinasi Audio Stage 3 + Text Stage 4), bukan dua checkpoint terpisah.
   - Tulis deskripsinya menjadi:
     - **Bimodal Text Only:** Checkpoint `Bimodal Final` (3 seed), cabang audio dimatikan saat inferensi (`L_pred = 0`).
     - **Bimodal Audio Only:** Checkpoint `Bimodal Final` (3 seed), cabang teks dimatikan saat inferensi (`r_pred = 0`).

3. **Kunci *Cache Key* yang lengkap untuk Model Caching:**
   - Jangan hanya memakai pasangan `(lr, wd)` sebagai kunci cache.
   - Pastikan cache key mencakup: `(protocol_version, encoder, mode, seed, max_epochs, all_hyperparams)` termasuk parameter yang dikunci (`dropout_audio`, `dropout_text`, `lr_head`, `effective_batch`).
   - Ini mencegah tabrakan cache antar-mode (`audio_only` vs `bimodal`), antar-seed, atau terbaca file lama bekas 7 epoch.

4. **Catatan metodologis yang wajib ditulis di laporan:**
   - **Max 5 epoch (semua stage):** Secara fairness sudah tepat karena semua model mendapat budget komputasi yang sama. Namun, pantau kurva validation per epoch untuk `cnn_scratch` karena model *from scratch* berpotensi belum konvergen penuh di 5 epoch dibanding encoder pretrained (`ast`, `mert`, `beats`). Cukup dokumentasikan ini sebagai keterbatasan *fixed epoch budget*.
   - **2 knob per cabang (`dropout` dikunci):** Pastikan tidak ada syarat eksternal (misal rubrik tugas) yang mewajibkan minimal 3 hyperparameter per cabang; kalau tidak ada, 2 knob ini justru membuat OFAT sangat rapi (menguji 4 dari 6 titik grid audio dan 5 dari 9 titik grid teks).
   - **Bila pemenang Stage 1 berbeda encoder dengan pemenang Stage 4:** Tetapkan aturannya sejak awal, apakah `Unimodal Audio` di test set mengambil pemenang terbaik Stage 1 (*best unimodal vs best bimodal*), atau ditambah juga `Unimodal Audio` dari encoder pemenang Stage 4 (*controlled ablation* pada encoder yang sama).

Jika keempat catatan kecil di atas sudah kamu setujui, desain eksperimen ini sudah **final dan siap diimplementasikan** ke notebook.