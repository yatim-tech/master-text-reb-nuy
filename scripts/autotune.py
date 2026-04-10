import argparse
import os
import json
import logging
import glob
import re
from core.models.utility_models import (
    FileFormat,
    InstructTextDatasetType,
    DpoDatasetType,
    GrpoDatasetType,
)
from job_handler import create_job_text, start_tuning_container

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths that match text_trainer.py conventions (inside the training environment)
SOUTPUTS_DIR = "/workspace/scripts/soutputs"
LOG_DIR = "datasets"  # ds_folder in text_trainer.py


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------

def _find_trainer_state(output_dir: str) -> str | None:
    """
    Locate trainer_state.json written by HuggingFace Trainer.
    text_trainer.py appends '_N' to output_dir for each run attempt,
    so we look in output_dir and also in output_dir_0, output_dir_1, etc.
    """
    candidates = [output_dir] + sorted(
        glob.glob(f"{output_dir}_*"),
        key=lambda p: int(p.rsplit("_", 1)[-1]) if p.rsplit("_", 1)[-1].isdigit() else 0,
        reverse=True,  # prefer latest run
    )
    for base in candidates:
        # Direct file in base dir
        direct = os.path.join(base, "trainer_state.json")
        if os.path.exists(direct):
            return direct
        # Inside a checkpoint-N subdir (take the highest checkpoint)
        ckpt_files = sorted(
            glob.glob(os.path.join(base, "checkpoint-*", "trainer_state.json")),
            key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)),
        )
        if ckpt_files:
            return ckpt_files[-1]
    return None


def _parse_trainer_state(state_path: str, task_type: str) -> float:
    """
    Read trainer_state.json and return the best evaluation score.
    - Instruct / DPO : minimise  eval_loss     -> return min(eval_loss)
    - GRPO           : maximise  eval_reward   -> return -max(eval_reward)
                                                  (negated so the caller can always minimise)
    Returns float('inf') if no metric is found.
    """
    try:
        with open(state_path, "r") as f:
            state = json.load(f)
        log_history = state.get("log_history", [])
        if task_type == "grpo":
            values = [e["eval_reward"] for e in log_history if "eval_reward" in e]
            if values:
                best = max(values)
                logger.info(f"[trainer_state] eval_reward={best:.6f}  path={state_path}")
                return -best  # negate so caller can minimise
        else:
            values = [e["eval_loss"] for e in log_history if "eval_loss" in e]
            if values:
                best = min(values)
                logger.info(f"[trainer_state] eval_loss={best:.6f}  path={state_path}")
                return best
    except Exception as exc:
        logger.warning(f"Could not parse {state_path}: {exc}")
    return float("inf")


def _parse_log_file(log_path: str, task_type: str) -> float:
    """
    Fallback: grep the raw training log for eval metric lines emitted by
    HuggingFace Trainer, e.g.:
        {'eval_loss': 1.2345, 'eval_runtime': 12.3, ...}
        {"eval_reward": 0.87, ...}
    Returns float('inf') if nothing is found.
    """
    if not os.path.exists(log_path):
        return float("inf")

    metric_key = "eval_reward" if task_type == "grpo" else "eval_loss"
    higher_is_better = task_type == "grpo"
    pattern = re.compile(
        rf"""['\"]?{re.escape(metric_key)}['\"]?\s*:\s*"""
        r"""([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"""
    )
    values = []
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                for m in pattern.finditer(line):
                    try:
                        values.append(float(m.group(1)))
                    except ValueError:
                        pass
    except Exception as exc:
        logger.warning(f"Could not read log {log_path}: {exc}")
        return float("inf")

    if not values:
        return float("inf")

    best = max(values) if higher_is_better else min(values)
    logger.info(f"[log_parse] {metric_key}={best:.6f}  path={log_path}")
    return -best if higher_is_better else best


def extract_eval_metric(iter_task_id: str, task_type: str) -> float:
    """
    Extract the best eval metric for a completed dry-run, using:
      1. trainer_state.json  (primary, most reliable)
      2. raw log file        (fallback)
    Returns float('inf') on failure.
    """
    output_dir = os.path.join(SOUTPUTS_DIR, iter_task_id)
    log_path   = os.path.join(LOG_DIR, f"train_{iter_task_id}.log")

    # --- Primary: trainer_state.json ---
    state_path = _find_trainer_state(output_dir)
    if state_path:
        score = _parse_trainer_state(state_path, task_type)
        if score != float("inf"):
            return score

    # --- Fallback: raw log ---
    score = _parse_log_file(log_path, task_type)
    if score != float("inf"):
        return score

    logger.warning(
        f"No eval metric found for {iter_task_id}. "
        f"Checked: {output_dir}*, {log_path}"
    )
    return float("inf")


# ---------------------------------------------------------------------------
# Main autotune loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autotune hyperparameters for Master-Text")
    parser.add_argument("--task_id",    type=str, required=True)
    parser.add_argument("--task_type",  type=str, required=True,
                        choices=["instruct", "dpo", "grpo"])
    parser.add_argument("--model",      type=str, required=True, help="HF Model ID")
    parser.add_argument("--dataset",    type=str, required=True, help="Path or HF dataset ID")
    parser.add_argument("--file_format",type=str, default="json",
                        choices=["json", "hf"], help="Dataset file format")
    args = parser.parse_args()

    # Hyperparameter search grid
    lrs         = [3e-6, 5e-6, 8e-6]
    batch_sizes = [4, 8, 16]
    vllm_mems   = [0.4, 0.6, 0.8] if args.task_type == "grpo" else [None]

    dataset_type_map = {
        "instruct": InstructTextDatasetType(),
        "dpo":      DpoDatasetType(),
        "grpo":     GrpoDatasetType(),
    }
    dataset_type     = dataset_type_map[args.task_type]
    file_format_enum = FileFormat.JSON if args.file_format == "json" else FileFormat.HF

    best_score  = float("inf")
    best_params = {}
    results     = []

    for lr in lrs:
        for bs in batch_sizes:
            for mem in vllm_mems:
                iter_task_id = f"{args.task_id}_{lr}_{bs}"
                logger.info(
                    f"\n{'='*60}\n"
                    f"DRY RUN  LR={lr}  BS={bs}  VLLM_MEM={mem}\n"
                    f"{'='*60}"
                )

                # Pass hyperparameters via env vars (picked up by DockerEnvironment)
                os.environ["AUTOTUNE_LR"]         = str(lr)
                os.environ["AUTOTUNE_BATCH_SIZE"] = str(bs)
                if mem is not None:
                    os.environ["AUTOTUNE_VLLM_MEM"] = str(mem)

                job = create_job_text(
                    job_id=iter_task_id,
                    dataset=args.dataset,
                    model=args.model,
                    dataset_type=dataset_type,
                    file_format=file_format_enum,
                    expected_repo_name="autotune-dry-run",
                )

                try:
                    start_tuning_container(job)
                    score = extract_eval_metric(iter_task_id, args.task_type)
                except Exception as exc:
                    logger.error(f"Dry run failed (OOM or crash): {exc}")
                    score = float("inf")

                record = {
                    "lr": lr,
                    "batch_size": bs,
                    "vllm_gpu_memory_utilization": mem,
                    "score": score,
                }
                results.append(record)
                logger.info(f"Score = {score:.6f}  (lower is always better here)")

                if score < best_score:
                    best_score  = score
                    best_params = {"lr": lr, "batch_size": bs,
                                   "vllm_gpu_memory_utilization": mem}

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "="*60 + "\nAUTOTUNE FINISHED\n" + "="*60)
    logger.info("Full results (sorted best → worst):")
    for r in sorted(results, key=lambda x: x["score"]):
        logger.info(
            f"  LR={r['lr']:<10}  BS={r['batch_size']:<4}  "
            f"VLLM_MEM={r['vllm_gpu_memory_utilization']}  "
            f"score={r['score']:.6f}"
        )
    logger.info(f"\nBEST CONFIG : {best_params}  (score={best_score:.6f})")

    out_path = f"/dataset/outputs/optimal_param_{args.task_id}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({**best_params, "best_score": best_score}, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
