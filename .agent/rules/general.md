# Aturan Umum (General Rules) Repositori Master-Text

Ini adalah repositori sentral untuk proses Pelatihan (Training) dan Penyelarasan (Fine-Tuning) *Large Language Models (LLM)* tingkat lanjut milik pengguna. Repositori ini mendukung metode **Supervised Fine-Tuning (Instruct), Direct Preference Optimization (DPO),** dan **Generative Reward Policy Optimization (GRPO)**.

Saat beroperasi di repositori ini, pastikan Anda sebagai entitas kode mengikuti arsitektur dan hukum logika di bawah ini agar perubahan/skrip yang Anda sarankan tidak asala-asalan.

---

## 1. Arsitektur Folder Wajib Diketahui
- **`scripts/`**: INTI REPOSITORI. Semua eksekusi komputasi keras, skrip pelatihan utama (`train_*.py`), definisi parameter dinamis (`*_config.py`), dan skrip *preprocessing/tokenization* bersarang di sini. Modifikasi komputasi LLM dilakukan di sini.
- **`trainer/`**: Berfungsi sebagai antarmuka API/Server Backend yang menjembatani server (via ASGI/endpoints) dengan antrean tugas asinkron (`tasks.py`).
- **`dockerfiles/`**: Berisi manifest/fondasi lingkungan kontainer (seperti `standalone-text-trainer.dockerfile`). Asumsikan semua eksekusi Python yang aslinya berjalan akan dilakukan **di dalam selubung Docker**, bukan di lingkungan murni sistem OS host.
- **`mcp-auto-tuner/`**: Ekstensi server lokal kustom berbasis Model Context Protocol (MCP) untuk memfasilitasi *auto-tuning hyperparameter* oleh AI.

---

## 2. Aturan Merombak Parameter Konfigurasi (Tuning)
JANGAN PERNAH menyarankan pengguna mengubah kode konfigurasi (seperti nilai asal *Learning Rate* atau *Batch Size*) menggunakan metode tebak-tebakan. Ikuti hukum hierarki *Master-Text* berikut saat membedah `instruct_config.py`, `dpo_config.py`, atau `grpo_config.py`:

1. **Titik Awal (Dictionary Mapping)**: Cari referensi parameter utama berdasarkan rentang jumlah parameter (*Billion parameters*) model (misal kunci kamus `0_1_b`, `4_5_b`, `40_80_b`).
2. **Kewaspadaan Terhadap Hardcode Overrides (Sangat Penting)**: Jangan sekadar mengubah tabel *dictionary* bagian atas! Selalu baca hingga baris terbawah (di dalam fungsi `get_training_json()` atau sejenisnya). Skrip ini cerdas dan sangat sering secara sengaja **memotong (*slice*)/meng-override nilai `batch_size` dari dictionary ke persentase mutlak** untuk menangani kerentanan memori pada arsitektur tertentu (contoh: model `falcon`, `phi`, `gpt-j`, `gemma-2-9b` akan selalu dipotong paksa).
3. **Penyelamatan VRAM (*Memory Optimization*)**:
   - Jika `OOM` (*Out of Memory*) pada **Instruct**: Ingat untuk selalu mencoba menonaktifkan mode pemampatan data (`--packing False`) sebelum memotong `batch_size`.
   - Jika `OOM` pada **GRPO**: Evaluasi ulang proporsi memory peladen *Student-Teacher* lewat mengatur parameter `--vllm_gpu_memory_utilization` atau kurangi tebaran paralel generasi sel serentak (`--num_generations`). VRAM GRPO dieksploitasi dengan sangat ekstrim.
   - Pahami bahwa fitur `gradient_accumulation_steps` mayoritas dihitung matematis secara terotomatisasi di ujung skrip (misal: `int(64 / total_batch_size)` agar menyentuh angka aman 64). Jangan memaksa mendeklarasikannya secara paksa dalam tabel utama jika berlawanan dengan rumus ujung ini.

---

## 3. Pengecualian Tabel Eksternal (LRS Lookup)
Bila dalam kode akhir skrip membaca variabel saklar **`find_lk_lr` bernilai `True`**, seluruh keringat modifikasi Anda terhadap *Learning Rate (LR)* di dalam file konfigurasi akan lenyap tak berguna! Anda (AI) WAJIB menyadari bahwa skrip tersebut **memprioritaskan nilai eksternal dari fungsi yang berafiliasi di dalam `lrs_lookup.py`** (contoh fungsi: `get_instruct_lr`, `get_grpo_python_lr`, atau `get_dpo_lr`).  
> **Aksi:** Jika ini terjadi, ubah/timpa nilai keluaran yang bersumber dari tabel file `lrs_lookup.py` tersebut secara langsung, bukan malah mengotak-atik file konfigurasi utamanya.

---

## 4. Evaluasi & Logging
- **Deadline Komputasi (Early Stopping)**: Evaluasi pelatihan berjalan bukan dari sistem dasar Huggingface murni, melainkan melewati `CustomEvalSaveCallback`. Skrip *training* dibangun dengan kemampuan untuk **mendeteksi *deadline* sisa waktu** komputasi sewa. Pelatihan bisa berhenti dengan apik jika waktunya akan habis.
- **Tracking Log & Metrik**: Monitoring tren grafik *Loss/Reward* terintegrasi ke `wandb` (Weights & Biases). Saat AI Anda memanggil alat penganalisa log/Server MCP, arahkan pemonitoran ke ekstraksi nilai file `trainer_state.json` lokal.

---

## 5. Aturan Turunan dari Commit Terbaru (Crystallized Rules)
Bagian ini merangkum aturan-aturan yang tersebar di kode & commit recent. Letakkan di sini agar semua *skill* (autotune-*, diagnose-regression, master-tuner) merujuk pada sumber yang sama dan tidak drift.

### 5.1 Target Effective Batch Size (Instruct)
Dihitung otomatis di `get_training_json` — **jangan paksakan** via tabel atas.
- `<2B` → `24` (memaksimalkan jumlah gradient update untuk model kecil full-FT).
- `2-4B` → `32` (seimbang).
- `≥4B` → `64` (gradient stabil untuk model besar).
`gradient_accumulation_steps = target_effective_bs / (batch_size × gpu_nums)` bila agregat per-step `< target`.

### 5.2 Aturan Optimizer
- **Instruct `<4B`**: `adamw_torch_fused` (full-precision fused, lebih akurat, VRAM masih aman di skala ini).
- **Instruct `≥4B`**: `paged_adamw_8bit` (+ LoRA by default; VRAM ketat).
- **DPO (semua ukuran)**: `paged_adamw_8bit`. Jangan fallback ke `adamw_torch_fused`.
- **GRPO (semua ukuran)**: `paged_adamw_8bit`.

### 5.3 LR Cap untuk Instruct `<4B` Full-FT
Bila `find_lk_lr=True` dan lookup LR dari [scripts/lrs/instruct.json](scripts/lrs/instruct.json) `> 3 × config_lr`, **cap di `3 × config_lr`**. Alasan: tabel lookup dihitung di pipeline lama dengan batch besar + packed eval, tidak kompatibel dengan pipeline sekarang. Referensi: [scripts/instruct_config.py](scripts/instruct_config.py) fungsi `get_training_json`.

### 5.4 LR Scheduler
- Tipe: `cosine_with_restarts` (SGDR), **bukan** `cosine_with_min_lr` (sudah diganti di commit `22572bd`).
- `num_cycles = min(epoch_num, 3)` — aligned ke epoch boundary.
- `warmup_ratio = 0.05`.
- `epoch_num` *time-budget aware* (lihat `_get_epoch_num` di masing-masing `*_config.py`); mempengaruhi bentuk kurva LR, bukan sekadar durasi.

### 5.5 Don't-Regress List
Aturan yang sudah "berdarah" untuk ditemukan. Jangan ulangi:
- **Instruct `neftune_noise_alpha`**: harus tetap `0`. Reintroduksi regularization agresif pernah menaikkan `eval_loss` ke 2.514 (precedent: commit `641ccd1`).
- **Instruct dev dataset `<4B`**: jangan di-*pack*. Precedent: commit `bbeef16`.
- **Instruct `<2B`**: butuh intensitas tinggi (LR penuh, epoch penuh, min_steps tinggi). Jangan turunkan intensitas. Precedent: commit `ac12b67`.
- **Triton init**: harus terjadi sebelum `transformers` di-*import* (CUDA init dulu). Precedent: commit `59721d0`, `0492b3f`, `e6125b4`. Metode saat ini: stub `sys.modules` sebelum import.
- **GRPO `allow_find_lk_lr`**: di-*hardcode* `False` — jalur lookup nonaktif by default. Jangan asumsikan lookup jalan kecuali flag dihidupkan.

---

> **Tugas Akhir untuk AI:**
> *Dalam menganalisis *bug/error traceback*, agen pemrograman AI wajib melacak alur eksekusi dari hulu ke hilir (Contoh: Parameter JSON > `*config.py` > `text_trainer.py` > eksekusi akhir loop di `train_*.py`) sebelum melontarkan asumsi perbaikan kodingan ke repositori pengguna.*
