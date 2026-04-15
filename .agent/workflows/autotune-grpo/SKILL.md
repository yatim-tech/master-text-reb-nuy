---
name: autotune-grpo
description: Alur Kerja (Workflow) Otonom AI untuk Men-Tuning Hyperparameter LLM pada jalur Generative Reward Policy Optimization (GRPO). Fokus pada kalibrasi vLLM memory, num_generations, reward function, dan dry run MCP.
---

## Skenario Utilitas
Gunakan *skill* ini saat Pengguna memerintahkan tuning konfigurasi untuk pelatihan **GRPO**. Untuk SFT atau preferensi, beralih ke `/autotune-instruct` atau `/autotune-dpo`.

## Prasyarat Lingkungan Kerja
Tergantung pada `AutoTuner_MasterText` MCP Server (`.agent/mcp-auto-tuner/server.py`).

## Sumber Kebenaran Konfigurasi
- Tabel bucket: [scripts/grpo_config.py](scripts/grpo_config.py) — dict `GRPO_CONFIG` dan fungsi `get_grpo_config(param_nums)`.
- Tabel LR luar (non-python reward): [scripts/lrs/grpo.json](scripts/lrs/grpo.json) via `get_grpo_lr(model_name)`.
- Tabel LR luar (python reward): [scripts/lrs/grpo_python.json](scripts/lrs/grpo_python.json) via `get_grpo_python_lr(model_name)`.
- Detektor reward: `if_contain_slow_reward_function()`, `contain_python_execution()` di [scripts/grpo_config.py](scripts/grpo_config.py).

## Langkah-langkah Eksekusi

1. **Pemetaan Metadata**
   - Tentukan `param_nums`.
   - Bucket GRPO: `0_1_b`, `1_2_b`, `2_4_b`, `4_5_b`, `5_6_b`, `6_9_b`, `9_12_b`, `12_15_b`, `15_20_b`, `20_40_b`, `40_80_b`.
   - Periksa **tipe reward function** di `train_info["dataset_type"]`:
     - Mengandung `langcheck` / `detoxify` / `textstat` → *slow reward* → `batch_size` akan di-*override* oleh `get_training_json` (lihat blok `if if_contain_slow_reward_function`).
     - Mengandung `sat_reward_function` / `ded_reward_function` / `abd_reward_function` → python execution → LR lookup beralih ke `grpo_python.json`.

2. **Eksekusi Dry Run**
   - `run_training_trial --max_steps 100`.
   - GRPO adalah task paling lambat per-step: setiap step membangkitkan `num_generations` sampel + menghitung reward. Ekspektasi throughput jauh lebih rendah dari SFT/DPO.

3. **Penawar OOM (Urut Prioritas untuk GRPO — BERBEDA dari SFT/DPO)**
   1. **Pertama**: turunkan `vllm_gpu_memory_utilization` (default per bucket `0.35` – `0.8`). Ini memberi kembali VRAM ke policy model tanpa mengurangi throughput generasi.
   2. **Kedua**: kurangi `num_generations` (default `8` untuk `<2B`, turun ke `2` untuk `>12B`). Ingat: `num_generations` minimum realistis `= 2` (untuk komparasi reward).
   3. **Ketiga**: belah dua `batch_size`.
   4. Untuk model `>15B`, `use_vllm=False` sudah default — jangan paksa `True`.
   5. Untuk model `>20B`, `use_4bit=True` sudah default — mempertahankannya wajib.

4. **Monitoring Reward & Kalibrasi**
   - Metrik utama: `eval_reward` (bukan `eval_loss`); `greater_is_better=True`, `metric_for_best_model=eval_reward`.
   - Tarik via `check_wandb_run`. Awasi plateau reward dan divergensi policy.
   - **Catatan pencarian LR via lookup**: saat ini `allow_find_lk_lr = False` di [scripts/grpo_config.py](scripts/grpo_config.py) — jalur lookup DINONAKTIFKAN. LR yang dipakai = `config.lr * reg_ratio`. Jangan mengandalkan `find_lk_lr` untuk GRPO kecuali flag itu dihidupkan dulu.
   - Gunakan `extend_learning_rates` (`log_range=0.2`) untuk eksplorasi bounded dari baseline bucket.

5. **Catatan Karakteristik GRPO**
   - Optimizer selalu `paged_adamw_8bit`.
   - `weight_decay = 0.005`.
   - Model `starcoder*`: `batch_size` otomatis `/1.5`.
   - Model `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`: `use_lora` dipaksa `True`.
   - `save_total_limit = 2` (lebih ketat dari SFT/DPO).
   - `eval_batch_size = 4` (atau `2` bila `batch_size <= 4`).
   - Override env: `AUTOTUNE_LR`, `AUTOTUNE_BATCH_SIZE`, `AUTOTUNE_VLLM_MEM` semua bekerja di akhir `get_training_json`.

6. **Dokumentasi & Deploy**
   - Tabel Markdown: bucket, LR akhir, `vllm_gpu_memory_utilization`, `num_generations`, `use_vllm`, `use_4bit`, effective batch size, epoch_num.
   - Sertakan grafik `train/reward` dari W&B.
   - Unggah lewat `upload_to_huggingface` setelah Pengguna menyetujui.
