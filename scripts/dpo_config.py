from model_utility import get_model_architecture, get_model_num_params, get_use_liger, disable_flash_attention, get_gradient_checkpointing, get_gpu_count
from copy import deepcopy
from lrs_lookup import get_dpo_lr

DPO_CONFIG = {
    "0_1_b": {
        "lr": 1.35e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 16,
        "beta": 0.05,
        "save_before_remaining_time": 3,
    },
    "1_2_b": {
        "lr": 8.7e-6,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 12,
        "beta": 0.05,
        "save_before_remaining_time": 3,
    },
    "2_4_b": {
        "lr": 6.5e-6,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 12,
        "use_lora": True,
        "beta": 0.05,
        "save_before_remaining_time": 3,
    },
    "4_5_b": {
        "lr": 6.25e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 12,
        "use_lora": True,
        "beta": 0.1,
        "save_before_remaining_time": 4,
    },
    "5_9_b": {
        "lr": 7.5e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 8,
        "use_lora": True,
        "beta": 0.1,
        "save_before_remaining_time": 5,
    },
    "9_12_b": {
        "lr": 5e-6,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 32,
        "gradient_checkpointing": False,
        "beta": 0.1,
        "save_before_remaining_time": 8,
    },
    "12_14_b": {
        "lr": 8.5e-6,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 24,
        "gradient_checkpointing": False,
        "beta": 0.15,
        "save_before_remaining_time": 10,
    },
    "14_15_b": {
        "lr": 8.5e-6,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 18,
        "gradient_checkpointing": False,
        "beta": 0.15,
        "save_before_remaining_time": 12,
    },
    "15_40_b": {
        "lr": 8e-6,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 16,
        "gradient_checkpointing": False,
        "beta": 0.2,
        "save_before_remaining_time": 12,
    },
    "40_80_b": {
        "lr": 8e-6,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 8,
        "gradient_checkpointing": False,
        "beta": 0.2,
        "save_before_remaining_time": 15,
    },
}

for key in DPO_CONFIG:
    DPO_CONFIG[key]["label"] = key
    

def get_config(param_nums: int) -> dict:
    result = None
    if param_nums < 1_000_000_000:
        result = DPO_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        result = DPO_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        result = DPO_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        result = DPO_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        result = DPO_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        result = DPO_CONFIG["9_12_b"]
    elif param_nums < 14_000_000_000:
        result = DPO_CONFIG["12_14_b"]
    elif param_nums < 15_000_000_000:  
        result = DPO_CONFIG["14_15_b"]
    elif param_nums < 35_000_000_000:
        result = DPO_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        result = DPO_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported", flush=True)
        result = {
            "lr": 4e-5,
            "distributed": "ds",
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True
        }
    if param_nums < 4_000_000_000 and param_nums > 1_330_000_000:
        result["gpu_count"] = 2
    if param_nums > 13_330_000_000:
        result["gpu_count"] = 8
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
        "beta",
        "use_liger",
        "optimizer",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")
    gpu_nums = get_gpu_count()
    start_cmd = "python"
    run_type = config.get("distributed", "ddp")
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_dpo.py \
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
    --beta {beta} \
    --weight_decay {weight_decay} \
    --lr_scheduler_type cosine_with_restarts \
    --warmup_ratio 0.05 \
    --lr_scheduler_kwargs "{\\"num_cycles\\": {num_cycles}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} --disable_fa {disable_fa}"""
    )

    if config.get("use_lora", False):
        template += (
            " --use_peft --lora_r 128 --lora_alpha 256 --lora_target_modules all-linear"
        )

    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))
    
    if config.get("use_attn_implementation", ""):
        use_attn_implementation = config["use_attn_implementation"]
        template = template + f""" --use_attn_implementation {use_attn_implementation}"""
        
    return template


def _get_epoch_num(param_nums: int, hours: float) -> int:
    """
    DPO is slower than SFT per sample (reference model + chosen/rejected pair),
    so thresholds are slightly more conservative than Instruct.
    """
    if param_nums < 1_000_000_000:
        if hours < 2:    return 2
        if hours < 4:    return 3
        return 4
    elif param_nums < 4_000_000_000:
        if hours < 2:    return 1
        if hours < 5:    return 2
        return 3
    elif param_nums < 9_000_000_000:
        if hours < 3:    return 1
        if hours < 7:    return 2
        return 3
    elif param_nums < 15_000_000_000:
        if hours < 5:    return 1
        return 2
    else:
        return 1


def _get_periodic_save_steps(param_nums: int) -> int:
    """
    DPO is slower per step than Instruct (reference model forward pass),
    so fewer steps/hour -> can use slightly larger intervals than Instruct.
    Target: save every ~15-20 minutes.
    """
    if param_nums < 1_000_000_000:
        return 150
    elif param_nums < 4_000_000_000:
        return 100
    elif param_nums < 9_000_000_000:
        return 80
    elif param_nums < 15_000_000_000:
        return 50
    else:
        return 50


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    config = get_config(param_nums)
    run_config = {
        "epoch_num": _get_epoch_num(param_nums, train_info["hours_to_complete"]),
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "num_cycles": _get_num_cycles(_get_epoch_num(param_nums, train_info["hours_to_complete"])),
        "save_total_limit": max(2, _get_epoch_num(param_nums, train_info["hours_to_complete"])),
        "beta": config.get("beta", 0.1),
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.001,
        "use_attn_implementation": "kernels-community/vllm-flash-attn3" if train_info.get("is_openai", False) else ""
    }
    
    if not config.get("gradient_checkpointing", True):
        run_config["gradient_checkpointing"] = False
    
    total_batch_size = run_config["batch_size"] * run_config["gpu_nums"]
    if total_batch_size < 64:
        run_config["gradient_accumulation_steps"] = min(4, int(64 / total_batch_size))
    
    if train_info["find_lk_lr"]:
        lr = get_dpo_lr(model_name)
        if lr is not None:
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
    if run_config["disable_fa"] == "False":
        run_cmd = run_cmd + " --padding_free True"
    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = config.get("save_before_remaining_time", 3)
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = _get_periodic_save_steps(param_nums)
    train_request["checking_step"] = 80
    
    return {
        "train_request": train_request,
        "run_cmd": run_cmd
    }
