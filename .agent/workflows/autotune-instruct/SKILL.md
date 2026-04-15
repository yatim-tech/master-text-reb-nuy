---
name: autotune-instruct
description: Alur Kerja (Workflow) Otonom AI untuk Men-Tuning Hyperparameter LLM pada jalur Supervised Fine-Tuning (Instruct/SFT). Fokus pada kalibrasi LR, penanggulangan OOM, dan eksekusi dry run via MCP.
---

## Skenario Utilitas
Gunakan *skill* ini saat Pengguna memerintahkan tuning konfigurasi untuk pelatihan **Instruct (SFT)**. Untuk jalur preferensi atau reward, beralih ke `/autotune-dpo` atau `/autotune-grpo`.

## Prasyarat Lingkungan Kerja
Tergantung pada `AutoTuner_MasterText` MCP Server (`.agent/mcp-auto-tuner/server.py`).

## Sumber Kebenaran Konfigurasi
- Tabel bucket: [scripts/instruct_config.py](scripts/instruct_config.py) — dict `INSTRUCT_CONFIG` dan fungsi `get_instruct_config(param_nums)`.
- Tabel LR luar: [scripts/lrs/instruct.json](scripts/lrs/instruct.json) via `get_instruct_lr(model_name)` di [scripts/lrs_lookup.py](scripts/lrs_lookup.py).
- Utilitas pencarian LR bounded: [scripts/lr_utils.py](scripts/lr_utils.py) — `extend_learning_rates(lr, n, log_range=0.2)`.

## Langkah-langkah Eksekusi

1. **Pemetaan Metadata**
   - Tentukan `param_nums` model target.
   - Petakan ke bucket (`0_1_b`, `1_2_b`, `2_4_b`, `4_5_b`, `5_9_b`, `9_12_b`, `12_15_b`, `15_40_b`, `40_80_b`).
   - Jalankan `read_config` pada [scripts/instruct_config.py](scripts/instruct_config.py) dan telusuri `get_training_json` hingga akhir — bukan hanya tabel atas — karena ada *override* khusus (`falcon`, `phi`, `pythia`, `bloom`, `gptoss`, dll).

2. **Eksekusi Dry Run**
   - Gunakan `run_training_trial` dengan `--max_steps 100`.
   - Jangan lewati fase ini: *Full Run* tanpa dry run adalah pemborosan biaya sewa GPU.

3. **Penawar OOM (Urut Prioritas untuk Instruct)**
   1. Matikan `--packing` terlebih dahulu (aturan dari [.agent/rules/general.md](.agent/rules/general.md) §2.3).
   2. Jika masih OOM, belah dua `batch_size`.
   3. Jangan utak-atik tabel `INSTRUCT_CONFIG` langsung kalau masalahnya arsitektur spesifik — cek dulu apakah `get_training_json()` sudah memotongnya (`phi-2`, `phi-1_5` = bs/4; `pythia-160m` = bs/1.5; dll).

4. **Kalibrasi Learning Rate**
   - Tarik data eval via `check_wandb_run` atau `read_latest_eval_loss`.
   - Interpretasi:
     - *Loss* mengeras / stagnan → naikkan LR (gunakan `extend_learning_rates` dengan `log_range=0.2` untuk mengeksplor secara bounded).
     - *Loss* meledak / NaN → turunkan LR 10× kemudian eksplor bounded.
   - **LR Cap untuk <4B full FT**: jika `find_lk_lr=True` dan nilai lookup > `3 × config_lr`, cap di `3 × config_lr`. Alasan: tabel lookup dihitung di pipeline lama dengan batch besar + packed eval. Lihat [scripts/instruct_config.py:356-367](scripts/instruct_config.py).
   - Pakai `modify_config_regex` hanya untuk tambalan akut (bukan eksplorasi rutin).

5. **Catatan Don't-Regress Khusus Instruct**
   - `neftune_noise_alpha` harus tetap `0` untuk semua ukuran — jangan reintro-kan (precedent: commit `641ccd1` yang revert aggressive regularization sampai eval_loss regres ke 2.514).
   - Dev dataset `<4B` **jangan** di-*pack* (precedent: commit `bbeef16`).
   - Model `<2B` butuh intensitas pelatihan tinggi (min_steps & epoch tinggi) — jangan turunkan intensitas (precedent: commit `ac12b67`).
   - Efek `reg_ratio` dikali di akhir: `run_config["learning_rate"] *= train_info["reg_ratio"]`. Awasi saat debugging LR.

6. **Dokumentasi & Deploy**
   - Laporkan tabel Markdown: baseline bucket, overrides yang aktif, LR akhir, effective batch size, epoch_num, num_cycles.
   - Tunggu persetujuan Pengguna sebelum `Full Run`.
   - Bila lulus, unggah dengan `upload_to_huggingface`.
