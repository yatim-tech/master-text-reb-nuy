---
name: diagnose-regression
description: Alur Kerja Forensik AI untuk membedah regresi eval_loss / eval_reward pasca perubahan konfigurasi atau kode. Mengurut kandidat penyebab berdasarkan commit precedent, tanpa menjalankan trial baru.
---

## Skenario Utilitas
Pakai *skill* ini ketika:
- `eval_loss` naik setelah commit/seri commit tertentu (SFT/DPO).
- `eval_reward` turun atau stagnan tidak wajar (GRPO).
- *Loss spike* atau NaN muncul di tengah training yang sebelumnya stabil.
- Pengguna meminta "bedah kenapa angkanya jadi begini" tanpa ingin memicu trial MCP.

Berbeda dari `/autotune-*` yang memicu `run_training_trial`, *skill* ini **murni forensik**: baca kode, baca commit, baca log. Output: kandidat penyebab diurut dengan bukti.

## Prasyarat Input dari Pengguna
Minimal satu dari:
- Rentang commit (`commit_before`, `commit_after`) atau cabang yang berbeda.
- Angka sebelum/sesudah: `eval_loss` atau `eval_reward`.
- Nama model + task (instruct / dpo / grpo).

## Sumber Investigasi
- `git log --oneline [range]` + `git diff [range] -- scripts/`.
- [scripts/instruct_config.py](scripts/instruct_config.py), [scripts/dpo_config.py](scripts/dpo_config.py), [scripts/grpo_config.py](scripts/grpo_config.py).
- [scripts/lr_utils.py](scripts/lr_utils.py), [scripts/lrs_lookup.py](scripts/lrs_lookup.py), [scripts/lrs/*.json](scripts/lrs/).
- `trainer_state.json` lokal di `output_dir`, atau W&B history via `check_wandb_run`.

## Langkah-langkah Eksekusi

1. **Pengumpulan Baseline**
   - Ekstrak angka eval sebelum dan sesudah. Kalau hanya punya "sesudah", cari checkpoint sebelumnya di W&B / `trainer_state.json`.
   - Identifikasi task (instruct/dpo/grpo) dan ukuran model (`param_nums`) karena tabel aturan berbeda per task.

2. **Diff Analisis**
   - `git log --oneline` untuk rentang commit yang dicurigai.
   - `git diff` fokuskan ke file konfigurasi dan utility:
     - `scripts/*_config.py`
     - `scripts/lr_utils.py`, `scripts/lrs_lookup.py`, `scripts/lrs/*.json`
     - `scripts/train_*.py`, `scripts/text_trainer.py`
     - `scripts/monkeypatch.py`, `scripts/patcher.py`, `scripts/patch_remote.py`

3. **Pencocokan Precedent (Commit History Repo Ini)**
   Urutkan kandidat berdasarkan relevansi. Masing-masing memiliki *precedent* nyata di commit history — kutip commit hash saat melapor.

   | Gejala | Kandidat Penyebab | Precedent Commit | Aksi |
   |---|---|---|---|
   | `eval_loss` naik tajam (Instruct <4B) | Regularization terlalu agresif (NEFTune, WD tinggi) | `641ccd1` "Revert aggressive regularization that caused eval_loss regression to 2.514" | Revert ke baseline; pastikan `neftune_noise_alpha=0` |
   | `eval_loss` naik (Instruct <4B) + dev set kecil | Packing `dev_ds` aktif | `bbeef16` "Stop packing dev_ds and lower effective batch size for <4B models" | Nonaktifkan packing dev; sesuaikan effective BS |
   | Intensitas pelatihan `<2B` terasa kurang | LR atau epoch/min_steps diturunkan | `ac12b67` "Restore LR, increase training intensity for <2B instruct models" | Restore LR & naikkan intensitas |
   | LR search pakai metric salah | Bukan `eval_loss` yang jadi pembanding | `e22dac4` "use eval_loss for LR search comparison and lower min_lr_rate" | Pastikan pembanding `eval_loss` |
   | Kurva LR meluruh terlalu cepat/lambat | `num_cycles` atau scheduler salah | `22572bd` "replace cosine_with_min_lr with cosine_with_restarts (SGDR)" | Pastikan `cosine_with_restarts` + `num_cycles = min(epoch_num, 3)` |
   | Training crash di awal (Triton) | Triton init setelah transformers import | `59721d0` / `0492b3f` / `e6125b4` | Pastikan CUDA init dulu, atau stub `sys.modules` |
   | `find_lk_lr=True` tapi LR aneh (Instruct <4B) | Lookup LR melampaui cap 3× config_lr | Logika di `get_training_json` [scripts/instruct_config.py:356-367](scripts/instruct_config.py) | Verifikasi nilai sudah ter-*cap* |
   | LR DPO/GRPO salah skala | Salah baca tabel (DPO `10⁻⁶`, GRPO `10⁻⁶`, SFT `10⁻⁵`) | — | Cocokkan dengan bucket tabel masing-masing |
   | GRPO reward plateau | `num_generations` terlalu rendah atau beta reward tidak sesuai | — | Cek `num_generations`, periksa tipe reward function |
   | Periodic save hilang | `periodic_save_steps` di-*hardcode* salah | `3cadef2` "Fix 3 issues from previous changes: periodic saves, checkpoint avg, LR lookup" | Verifikasi `_get_periodic_save_steps` untuk bucket |

4. **Verifikasi Kode Aktif (Jangan Percaya Memori)**
   - Untuk setiap kandidat, **baca file aktualnya**, jangan mengandalkan asumsi. Config sering di-*override* di `get_training_json()` atau di blok akhir fungsi.
   - Trace hulu-ke-hilir: `train_info JSON` → `*_config.py::get_training_json()` → `run_cmd` → `train_*.py` → loop di `text_trainer.py`.

5. **Rekomendasi Tindakan**
   - Kalau satu kandidat cocok precedent: arahkan revert spesifik (atau *cherry-pick* bagian yang diinginkan), sertakan commit hash revert.
   - Kalau kandidat baru (tidak ada precedent): rancang eksperimen **bounded**:
     - LR: `extend_learning_rates(current_lr, 3, log_range=0.2)` dari [scripts/lr_utils.py](scripts/lr_utils.py) → 3 titik di sekitar baseline.
     - Beta (DPO): ±0.05 dari nilai bucket.
     - `num_generations` (GRPO): +/- 1 langkah dari default bucket.
   - Jangan rekomendasi *random tweak* yang tidak bisa direproduksi.

6. **Output Laporan**
   Format tabel Markdown untuk setiap kandidat:
   ```
   | # | Hipotesis | Bukti (file/commit) | Keyakinan | Aksi |
   ```
   - Keyakinan: `Tinggi` (cocok precedent + diff cocok), `Sedang` (cocok salah satu), `Rendah` (asumsi).
   - Tutup dengan satu rekomendasi tindakan prioritas.

## Larangan
- Jangan memicu `run_training_trial` — itu ranah `/autotune-*`.
- Jangan menebak LR/beta/num_generations baru di luar eksperimen bounded.
- Jangan melakukan `git revert` atau edit file sebelum Pengguna menyetujui laporan forensik.
