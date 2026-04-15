---
name: autotune-dpo
description: Alur Kerja (Workflow) Otonom AI untuk Men-Tuning Hyperparameter LLM pada jalur Direct Preference Optimization (DPO). Fokus pada kalibrasi LR, beta, penanganan reference model, dan dry run MCP.
---

## Skenario Utilitas
Gunakan *skill* ini saat Pengguna memerintahkan tuning konfigurasi untuk pelatihan **DPO**. Untuk jalur SFT atau reward, beralih ke `/autotune-instruct` atau `/autotune-grpo`.

## Prasyarat Lingkungan Kerja
Tergantung pada `AutoTuner_MasterText` MCP Server (`.agent/mcp-auto-tuner/server.py`).

## Sumber Kebenaran Konfigurasi
- Tabel bucket: [scripts/dpo_config.py](scripts/dpo_config.py) — dict `DPO_CONFIG` dan fungsi `get_config(param_nums)`.
- Tabel LR luar: [scripts/lrs/dpo.json](scripts/lrs/dpo.json) via `get_dpo_lr(model_name)` di [scripts/lrs_lookup.py](scripts/lrs_lookup.py).
- Utilitas pencarian LR bounded: [scripts/lr_utils.py](scripts/lr_utils.py).

## Langkah-langkah Eksekusi

1. **Pemetaan Metadata**
   - Tentukan `param_nums`.
   - Bucket DPO: `0_1_b`, `1_2_b`, `2_4_b`, `4_5_b`, `5_9_b`, `9_12_b`, `12_14_b`, `14_15_b`, `15_40_b`, `40_80_b`.
   - Telusuri `get_training_json` di [scripts/dpo_config.py](scripts/dpo_config.py) hingga akhir — perhatikan *override* `gpu_count` untuk rentang `1.33B < x < 4B` (dipaksa 2) dan `>13.33B` (dipaksa 8).

2. **Eksekusi Dry Run**
   - `run_training_trial --max_steps 100`.
   - DPO memuat *reference model* di samping policy model → footprint VRAM ~2× SFT dengan ukuran yang sama. Rencanakan dry run dengan ekspektasi itu.

3. **Penawar OOM (Urut Prioritas untuk DPO)**
   1. Kurangi `batch_size` (DPO tidak punya flag `--packing` yang setara).
   2. Jika `disable_fa=False` di output, `--padding_free True` sudah otomatis ditambahkan — jangan duplikasi manual.
   3. Dari bucket `2_4_b` ke atas, LoRA sudah aktif by default (`use_lora=True`). Jangan mematikannya untuk hemat VRAM — itu akan mengubah karakter pelatihan.

4. **Kalibrasi Learning Rate & Beta**
   - Tarik data via `check_wandb_run` / `read_latest_eval_loss`.
   - LR DPO jauh lebih kecil dari SFT (orde `10⁻⁶`). Jangan bandingkan secara harfiah dengan angka SFT.
   - **Beta** (KL penalty) ditetapkan per bucket:
     - `0_1_b` – `2_4_b`: `0.05` (lunak, model kecil lebih responsif terhadap preferensi).
     - `4_5_b` – `9_12_b`: `0.1` (standar).
     - `12_14_b` – `14_15_b`: `0.15`.
     - `15_40_b` – `40_80_b`: `0.2` (rapat, mencegah model besar divergen dari reference).
   - Reward margin yang terlalu lebar + eval_loss naik = beta terlalu rendah. Reward margin stagnan mendekati 0 = beta terlalu tinggi.
   - Cek `find_lk_lr=True` → nilai lookup diambil apa adanya (tidak ada *cap* seperti Instruct).

5. **Catatan Karakteristik DPO**
   - Optimizer selalu `paged_adamw_8bit` — tidak ada fallback ke `adamw_torch_fused` meski ukuran kecil.
   - `weight_decay = 0.001` (jauh lebih kecil dari SFT yang `0.01`). Jangan samakan.
   - `min_steps = 100` di-*hardcode*; early stopping tetap via `CustomEvalSaveCallback`.
   - `gradient_accumulation_steps` dihitung otomatis untuk menargetkan `total_batch_size = 64` bila agregat per-step `<64`.

6. **Dokumentasi & Deploy**
   - Tabel Markdown: bucket, LR akhir, beta, effective batch size, epoch_num.
   - Sertakan grafik `rewards/chosen` vs `rewards/rejected` dari W&B kalau tersedia.
   - Unggah lewat `upload_to_huggingface` setelah disetujui.
