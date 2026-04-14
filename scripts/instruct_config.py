from model_utility import (
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
    disable_flash_attention,
    get_data_size,
    get_gpu_count,
)
from copy import deepcopy
from lrs_lookup import get_instruct_lr


FIXED_BS_CONFIG = {
    "EleutherAI/gpt-neo-1.3B": {"batch_size": 36},
    "EleutherAI/gpt-neo-125m": {"batch_size": 48},
    "bigscience/bloom-560m": {"batch_size": 10},
    "facebook/opt-1.3b": {"batch_size": 38},
    "facebook/opt-350m": {"batch_size": 36},
    "facebook/opt-125m": {"batch_size": 48},
}

INSTRUCT_CONFIG = {
    "0_1_b": {
        "lr": 0.0001,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 32,
        "use_lora": False,
        "save_before_remaining_time": 3,
    },
    "1_2_b": {
        "lr": 1e-4,
        "distributed": "ddp",
        "gpu_count": 1,
        "use_lora": False,
        "batch_size": 24,
        "save_before_remaining_time": 3,
    },
    "2_4_b": {
        "lr": 7.5e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 48,
        "save_before_remaining_time": 3,
    },
    "4_5_b": {
        "lr": 7e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 40,
        "save_before_remaining_time": 4,
    },
    "5_9_b": {
        "lr": 3.5e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 28,
        "save_before_remaining_time": 5,
    },
    "9_12_b": {
        "lr": 1e-4,
        "distributed": "ddp",
        "gpu_count": 2,
        "use_lora": True,
        "batch_size": 32,
        "save_before_remaining_time": 6,
    },
    "12_15_b": {
        "lr": 1e-4,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 30,
        "save_before_remaining_time": 10,
    },
    "15_40_b": {
        "lr": 8e-5,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 18,
        "save_before_remaining_time": 12,
    },
    "40_80_b": {
        "lr": 8e-5,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 8,
        "save_before_remaining_time": 15,
    },
}

for key in INSTRUCT_CONFIG:
    INSTRUCT_CONFIG[key]["label"] = key


def get_instruct_config(param_nums: int) -> dict:
    result = {
        "lr": 4e-5,
        "distributed": "ds",
        "gpu_count": 8,
        "batch_size": 6,
        "use_lora": True,
    }
    if param_nums < 1_000_000_000:
        result = INSTRUCT_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        result = INSTRUCT_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        result = INSTRUCT_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        result = INSTRUCT_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        result = INSTRUCT_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        result = INSTRUCT_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:
        result = INSTRUCT_CONFIG["12_15_b"]
    elif param_nums < 35_000_000_000:
        result = INSTRUCT_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        result = INSTRUCT_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported")
    result = deepcopy(result)
    if param_nums < 9_000_000_000 and param_nums > 8_000_000_000:
        result["batch_size"] = int(2 * result["batch_size"] / 3)
    return result


def _get_num_cycles(epoch_num: int) -> int:
    """
    Number of cosine restart cycles, aligned to epoch boundaries.
    - 1 epoch  → 1 cycle (plain cosine, no restart)
    - 2 epochs → 2 cycles (restart once at midpoint)
    - 3+ epochs → 3 cycles max (diminishing returns beyond 3)
    """
    return min(epoch_num, 3)


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "num_cycles",
        "use_liger",
        "optimizer",
        "use_lora",
        "packing",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    gpu_nums = get_gpu_count()
    start_cmd = "python"
    run_type = config["distributed"]
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_instruct.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --metric_for_best_model eval_loss \
    --save_only_model True \
    --save_total_limit {save_total_limit} \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine_with_restarts \
    --warmup_ratio 0.05 \
    --lr_scheduler_kwargs "{\\"num_cycles\\": {num_cycles}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} \
    --packing {packing} --disable_fa {disable_fa}"""
    )
    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    if config["use_lora"]:
        template = template + """ --use_lora True"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    if config.get("use_attn_implementation", ""):
        use_attn_implementation = config["use_attn_implementation"]
        template = (
            template + f""" --use_attn_implementation {use_attn_implementation}"""
        )

    if config.get("neftune_noise_alpha", 0) > 0:
        template += f" --neftune_noise_alpha {config['neftune_noise_alpha']}"

    return template


def _get_epoch_num(param_nums: int, hours: float) -> int:
    """
    Estimate the number of training epochs that realistically fit within the
    time budget. epoch_num shapes the cosine LR schedule — setting it too high
    means the LR decays too slowly within the actual training window;
    too low means the LR collapses before training time runs out.
    The time-aware stopping (WhenToEvalHandler) handles actual early exits.
    """
    if param_nums < 1_000_000_000:
        if hours < 1.5:  return 2
        if hours < 3:    return 3
        return 5
    elif param_nums < 4_000_000_000:
        if hours < 1.5:  return 2
        if hours < 3:    return 3
        return 5
    elif param_nums < 9_000_000_000:
        if hours < 3:    return 1
        if hours < 6:    return 2
        return 3
    elif param_nums < 15_000_000_000:
        if hours < 4:    return 1
        return 2
    else:
        return 1


def _get_periodic_save_steps(param_nums: int) -> int:
    """
    Target: save every ~15-20 minutes of training.
    Smaller models run faster (more steps/hour) so need smaller intervals.
    Larger models are already slow per step, so larger intervals are fine.
    Instruct baseline (most steps/hour among all task types).
    """
    if param_nums < 1_000_000_000:
        return 30
    elif param_nums < 4_000_000_000:
        return 25
    elif param_nums < 9_000_000_000:
        return 100
    elif param_nums < 15_000_000_000:
        return 80
    else:
        return 50


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    config = get_instruct_config(param_nums)

    # Full-precision fused AdamW for full FT (<4B) — more accurate than
    # paged 8-bit Adam and has no VRAM issue at this scale.
    # Keep paged 8-bit for larger / LoRA models where VRAM is tight.
    if param_nums < 4_000_000_000:
        _optimizer = "adamw_torch_fused"
        _neftune_alpha = 0
    else:
        _optimizer = "paged_adamw_8bit"
        _neftune_alpha = 0

    run_config = {
        "epoch_num": _get_epoch_num(param_nums, train_info["hours_to_complete"]),
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "num_cycles": _get_num_cycles(_get_epoch_num(param_nums, train_info["hours_to_complete"])),
        "save_total_limit": max(2, _get_epoch_num(param_nums, train_info["hours_to_complete"])),
        "use_liger": get_use_liger(model_architecture),
        "optimizer": _optimizer,
        "neftune_noise_alpha": _neftune_alpha,
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "packing": "True",
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": "True",
        "gradient_accumulation_steps": 4,
        "use_attn_implementation": (
            "kernels-community/vllm-flash-attn3"
            if train_info.get("is_openai", False)
            else ""
        ),
    }

    if run_config["disable_fa"] == "True" or model_architecture.strip().lower() in [
        "optforcausallm"
    ]:
        run_config["packing"] = "False"

    if model_name in FIXED_BS_CONFIG:
        run_config["batch_size"] = FIXED_BS_CONFIG[model_name]["batch_size"]

    if model_architecture.strip().lower() in [
        "gptneoxforcausallm",
        "gptjforcausallm",
        "phiforcausallm",
        "falconforcausallm",
    ]:
        run_config["batch_size"] = int(run_config["batch_size"] // 2)
        if model_name == "EleutherAI/pythia-160m":
            run_config["batch_size"] = int(run_config["batch_size"] / 1.5)
        elif "pythia" in model_name.lower():
            run_config["batch_size"] = int(run_config["batch_size"] / 1.8)

    if model_name in ["microsoft/phi-2", "microsoft/phi-1_5"]:
        run_config["batch_size"] = int(run_config["batch_size"] / 4)

    if "bloom-560m" in model_name or "bloomz-560m" in model_name:
        run_config["batch_size"] = 8

    if model_name == "mistralai/Mistral-7B-v0.1":
        run_config["batch_size"] = int(3 * run_config["batch_size"] / 4)

    if "falcon" in model_name.lower():
        run_config["batch_size"] = int(run_config["batch_size"] / 2)

    # Target effective batch size: smaller = more gradient updates.
    # <2B: 24 (maximizes updates for small full-FT models)
    # 2-4B: 32 (balanced)
    # 4B+: 64 (stable gradients for larger models)
    if param_nums < 2_000_000_000:
        target_effective_bs = 24
    elif param_nums < 4_000_000_000:
        target_effective_bs = 32
    else:
        target_effective_bs = 64
    data_per_step = run_config["batch_size"] * run_config["gpu_nums"]
    if data_per_step >= target_effective_bs:
        run_config["gradient_accumulation_steps"] = 1
    else:
        run_config["gradient_accumulation_steps"] = int(target_effective_bs / data_per_step)

    if model_architecture.strip().lower() in ["gptossforcausallm"]:
        run_config["use_lora"] = False

    if train_info["find_lk_lr"]:
        lr = get_instruct_lr(model_name)
        if lr is not None:
            if param_nums < 4_000_000_000:
                # For <4B full FT: lookup LRs were computed under the old
                # pipeline (huge batch size, packed eval). Allow up to 3×
                # our config LR to give the LR search enough room.
                max_lr = run_config["learning_rate"] * 3
                if lr > max_lr:
                    print(f"Lookup lr={lr} too high for <4B, capping at {max_lr}", flush=True)
                    lr = max_lr
            print(f"Using lr from lk: {lr}", flush=True)
            run_config["learning_rate"] = lr
        else:
            print(f"Using lr from config: {run_config['learning_rate']}", flush=True)

    run_config["learning_rate"] *= train_info["reg_ratio"]

    import os
    if os.environ.get("AUTOTUNE_LR"):
        run_config["learning_rate"] = float(os.environ.get("AUTOTUNE_LR"))
    if os.environ.get("AUTOTUNE_BATCH_SIZE"):
        run_config["batch_size"] = int(os.environ.get("AUTOTUNE_BATCH_SIZE"))

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])
    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = config.get("save_before_remaining_time", 3)
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = _get_periodic_save_steps(param_nums)
    train_request["checking_step"] = 80 if param_nums < 4_000_000_000 else 100

    if param_nums < 1_000_000_000:
        train_request["min_steps"] = max(
            int(train_info["hours_to_complete"] * 100), train_request["min_steps"]
        )

    elif param_nums < 9_000_000_000:
        train_request["min_steps"] = max(
            int(train_info["hours_to_complete"] * 70), train_request["min_steps"]
        )

    return {"train_request": train_request, "run_cmd": run_cmd}
