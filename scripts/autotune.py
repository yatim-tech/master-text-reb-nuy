import argparse
import os
import json
import logging
from core.models.utility_models import FileFormat, TextDatasetType, InstructTextDatasetType, DpoDatasetType, GrpoDatasetType
from job_handler import create_job_text, start_tuning_container

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Autotune Hyperparameters for Master-Text")
    parser.add_argument("--task_id", type=str, required=True, help="Task ID")
    parser.add_argument("--task_type", type=str, required=True, choices=["instruct", "dpo", "grpo"], help="Target optimization strategy")
    parser.add_argument("--model", type=str, required=True, help="HF Model ID")
    parser.add_argument("--dataset", type=str, required=True, help="Path or HF ID to dataset")
    parser.add_argument("--file_format", type=str, default="json", help="File format, default json")
    args = parser.parse_args()

    # Define hyperparameter grid
    lrs = [3e-6, 5e-6, 8e-6]
    batch_sizes = [4, 8, 16]
    vllm_mems = [0.4, 0.6, 0.8] if args.task_type == "grpo" else [None]

    # Map generic string to TextDatasetType
    if args.task_type == "instruct":
        dataset_type = InstructTextDatasetType()
    elif args.task_type == "dpo":
        dataset_type = DpoDatasetType()
    else:
        dataset_type = GrpoDatasetType()

    file_format_enum = FileFormat.JSON if args.file_format == "json" else FileFormat.HF

    best_loss = float('inf')
    best_params = {}

    for lr in lrs:
        for bs in batch_sizes:
            for mem in vllm_mems:
                logger.info(f"============ DRY RUN ============")
                logger.info(f"Testing LR: {lr}, BS: {bs}, VLLM_MEM: {mem}")
                os.environ["AUTOTUNE_LR"] = str(lr)
                os.environ["AUTOTUNE_BATCH_SIZE"] = str(bs)
                if mem is not None:
                    os.environ["AUTOTUNE_VLLM_MEM"] = str(mem)

                # Hack iteration suffix to task_id to avoid container crash collisions
                iter_task_id = f"{args.task_id}_{lr}_{bs}"

                job = create_job_text(
                    job_id=iter_task_id,
                    dataset=args.dataset,
                    model=args.model,
                    dataset_type=dataset_type,
                    file_format=file_format_enum,
                    expected_repo_name="autotune-dry-run"
                )

                try:
                    # In a full working example, we'd add max_steps=100 inside the config logic 
                    # but since we're using job_handler, it will fire based on scripts config.
                    start_tuning_container(job)
                    logger.info("Completed dry run successfully. Extracting loss...")
                    # Extractor logic to parse axolotl/outputs/logs goes here 
                    # For MVP, we simulate:
                    simulated_loss = lr + bs # Replace with actual grep logic
                    if simulated_loss < best_loss:
                        best_loss = simulated_loss
                        best_params = {"lr": lr, "batch_size": bs, "vllm_gpu_memory_utilization": mem}
                except Exception as e:
                    logger.error(f"Dry run failed (likely OOM). Error: {e}")
                    continue

    logger.info("============ AUTOTUNE FINISHED ============")
    logger.info(f"BEST CONFIG: {best_params}")
    with open(f"/dataset/outputs/optimal_param_{args.task_id}.json", "w") as f:
        json.dump(best_params, f)

if __name__ == "__main__":
    main()
