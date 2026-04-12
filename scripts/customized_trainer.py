from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
from typing import Callable, Optional, Dict
import shutil
import json
from transformers.trainer_utils import is_main_process
import wandb
import torch
from state_manager import get_state, set_state
MAX_TRIES = 9


MIS_MATCH_VOCAB_SIZE_MODELS = [
    'NousResearch/Nous-Capybara-7B-V1',
    'berkeley-nest/Starling-LM-7B-alpha',
    'NousResearch/Hermes-2-Theta-Llama-3-8B',
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4'
]

ERROR_GENERATION_CONFIG_MODELS = [
    "lmsys/vicuna-7b-v1.5", 
    "lmsys/vicuna-13b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b", 
    "defog/llama-3-sqlcoder-8b"
]

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

print(f"LOCAL_RANK: {LOCAL_RANK} in customized_trainer.py", flush=True)
    
class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        max_steps: int = -1,
        checking_step: int = 100,
        total_steps_all_epochs: int = -1,
        end_time: str = "",
        checking_mode: str = "none"
    ):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        self.checking_step = checking_step
        self.total_steps_all_epochs = total_steps_all_epochs
        self.checking_mode = checking_mode
        self.end_time = end_time
        
    def compute_loss(self, state: TrainerState, metrics):
        return metrics.get("eval_loss", None)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Custom logic to decide whether to save or evaluate
        # print(f"************* on_step_end: {state.global_step}, check eval", flush=True)
        # TODO: implement the logic to save the model without evaluating if there is no check points --> avoid evaluating takes too much time
        # Check if the checking_step is reached
        # print(f"Checking the model at step: {state.global_step}, checking_step: {self.checking_step}, checking_mode: {self.checking_mode}", flush=True)
        if state.global_step == self.checking_step and self.checking_mode == "first_time":
            # print(f"Checking the model at step: {state.global_step}", flush=True)
            # check the time so far to estimate the training time in total 
            my_state = get_state()
            start_time_obj = datetime.datetime.strptime(my_state["train"]["start_time"], "%Y-%m-%d %H:%M:%S")
            start_train_time_obj = datetime.datetime.strptime(my_state["train"]["start_train_time"], "%Y-%m-%d %H:%M:%S")
            
            log_content = f"Checking the model at step: {state.global_step}"
            now = datetime.datetime.now()
            preparation_time = (start_train_time_obj - start_time_obj).total_seconds()
            log_content += f"\nPreparation time: {preparation_time}"
            time_so_far = (now - start_time_obj).total_seconds()
            log_content += f"\nTime so far: {time_so_far}"
            time_for_one_step = (now - start_train_time_obj).total_seconds() / self.checking_step
            log_content += f"\nTime for one step: {time_for_one_step}"
            # Now estimate the total training time for this training
            log_content += f"\nTotal steps all epochs: {self.total_steps_all_epochs}"
            total_remaining_training_time = time_for_one_step * (self.total_steps_all_epochs - state.global_step)
            log_content += f"\nTotal remaining training time: {total_remaining_training_time}"
            # n * time_so_far + total_remaining_training_time = total_remaining_time
            end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
            total_remaining_time = (end_time_obj - now).total_seconds()
            log_content += f"\nTotal remaining time: {total_remaining_time}"
            
            # n * time_so_far + (time_so_far + total_remaining_training_time) = total_remaining_time
            # time_so_far + total_remaining_training_time is the time it takes to finish the training (need to estimate the eval time and save time, assuming this is 15 minutes)
            # assuming time_so_far is + 5 minutes, just in case the checking step takes more time than expected
            max_var_time_sofar = 3 * 60
            n = (total_remaining_time - (time_so_far + total_remaining_training_time + 12 * 60)) / (time_so_far + max_var_time_sofar) # 300 = 5 minutes, assume that it extra time would be more or less 5 minutes
            n = int(n)
            my_state["check_details"] = {
                "now": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                "start_time": str(start_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "start_train_time": str(start_train_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "checking_step": self.checking_step,
                "checking_mode": self.checking_mode,
                "estimation_of_steps": n,
                "preparation_time": preparation_time,
                "time_so_far": time_so_far,
                "time_for_one_step": time_for_one_step,
                "total_remaining_training_time": total_remaining_training_time,
                "total_remaining_time": total_remaining_time,
                "end_time": self.end_time,
            }
            if n > 0: # we should try more 
                log_content += f"\nEstimated number of steps to complete the training: {n}"
                control.should_training_stop = True
                control.should_save = False
                args.save_strategy = "no"
                # Prefer eval_loss for fair comparison; fallback to training loss
                eval_entries = [e for e in state.log_history if "eval_loss" in e]
                if eval_entries:
                    my_state["train"]["current_loss"] = eval_entries[-1]["eval_loss"]
                else:
                    my_state["train"]["current_loss"] = state.log_history[-1].get("loss", float("inf"))
                my_state["mode"] = "continue"
                if n > MAX_TRIES:
                    n = MAX_TRIES
                log_content += f"\nFinal number: {n + 1}"
                my_state["next_runs"] = n + 1 # including the current run
            else:
                print(f"Time is not enough so we will finish the training", flush=True)
                my_state["mode"] = "finish"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)            
            return control
    
        elif state.global_step == self.checking_step and self.checking_mode == "second_time": # at second time, we don't estimate the training time again, just save the current_loss
            log_content = f"Checking the model at step: {state.global_step} where check_mode=second_time"
            my_state = get_state()
            # Use eval_loss from log history if available, fallback to training loss
            eval_entries = [e for e in state.log_history if "eval_loss" in e]
            if eval_entries:
                current_loss = eval_entries[-1]["eval_loss"]
            else:
                current_loss = state.log_history[-1].get("loss", float("inf"))
            my_state["train"]["current_loss"] = current_loss
                
            control.should_training_stop = True

            # Check if current_loss > current min_loss --> do not save to save time and space
            # 
            # if my_state["train"]["current_loss"] > current_min_loss:
            #     print(f"Current loss: {my_state['train']['current_loss']} is greater than the current min_loss: {current_min_loss}, do not save the checkpoint", flush=True)
            #     control.should_save = False
            # check if this is the last run and the current_loss is the lowest --> keep running the training
            current_is_the_best = False
            current_min_loss = min([run["current_loss"] for run in my_state["runs"]])
            if current_loss <= current_min_loss:
                if len(my_state["runs"]) + 1 == my_state["next_runs"]:
                    print(f"Current loss: {my_state['train']['current_loss']} is greater than: {current_min_loss}", flush=True)
                    current_is_the_best = True
                    
            if current_is_the_best:
                control.should_training_stop = False
                my_state["mode"] = "finish"
            else:
                control.should_save = False
                args.save_strategy = "no"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                # print(log_content, flush=True)
        
            
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            # do not allow the pod to be stopped by any reason 
                # first check if there is at least one checkpoint or not 
            print(f"Evaluating the model at step: {state.global_step} the reason: {when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                if not self.has_checkpoint: # if there is no checkpoint, we just save the model, do not evaluate
                    print(f"No checkpoint found, just save the model at step: {state.global_step}", flush=True)
                    control.should_evaluate = False
                    self.save_only = True
        return control


    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        self.save_only = False
        # Append eval_loss to file
        eval_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return 
        print(f"GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}", flush=True)
        if self.best_checkpoint_info is None or eval_loss < self.best_checkpoint_info["loss"]:
            print(f"Updating the best checkpoint info at step: {state.global_step} with eval_loss: {eval_loss}", flush=True)
            self.best_checkpoint_info = {
                "loss": eval_loss,
                "step": state.global_step
            }
            self.update_best_checkpoint = True
        else:
            if self.best_checkpoint_info is not None:
                print(f" At step: {state.global_step} The eval_loss: {eval_loss} is not smaller than the current best eval_loss: {self.best_checkpoint_info['loss']}, update_best_checkpoint={self.update_best_checkpoint}", flush=True)
            

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.global_step == self.max_steps and self.max_steps != -1:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            control.should_training_stop = True
        
        self.has_checkpoint = True
        
        if not is_main_process(LOCAL_RANK): # if not main process, skip this
            return 
            
        if self.save_only: # if only save, do not evaluate 
            print(f"Only save the model at step: {state.global_step}, no evaluation", flush=True)
            current_step = state.global_step
            # Remove existing directory if it exists
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
                
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{current_step}"),
                self.submission_dir
            )
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{current_step},no_eval")
            
            # release the flag
            self.save_only = False
            return 
            
        # Custom logic after model is saved
        # You can trigger external services, logs, or backups here
        if (
            self.update_best_checkpoint
            and is_main_process(LOCAL_RANK)
        ):
            print(f"Copy the best checkpoint to the submission directory at step: {state.global_step}", flush=True)
            # Remove existing directory if it exists
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
            best_eval_loss = self.best_checkpoint_info["loss"]
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}"),
                self.submission_dir
            )
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{self.best_checkpoint_info['step']},{best_eval_loss}")


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        eval_loss = None
        if state.log_history:
            last_log_entry = state.log_history[-1]
            eval_loss = last_log_entry.get("eval_reward", None)
            print(f"choose eval_loss ({eval_loss}) as eval_reward from: last_log_entry: {last_log_entry}; \n metrics: {metrics}", flush=True)
        else:
            print(f"state.log_history is empty", flush=True)
            
        if eval_loss is not None:
            eval_loss = - eval_loss
            
        return eval_loss
    
    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool: 
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)  # Make end_time timezone-aware in UTC
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"*** current time: {now} end_time: {end_time} time_diff: {time_diff}", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps

    def __call__(self, global_step: int) -> dict:
        
        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}
        
        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}
        
        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f"***ALERT: The time is about to run out need to eval & save the model", flush=True)
                # the eval time might be higher than the end_time, so we need to let the pod not stop by setting a flag for this
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}
        
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            return {"eval": True, "reason": "max_step"}

        return {"eval": False, "reason": "none"}


def set_generation_config(model_name, model):
    try:
        if model_name in ERROR_GENERATION_CONFIG_MODELS:
            model.generation_config = GenerationConfig(temperature=None, top_p=None)
    except:
        print(f"Error setting generation config for model {model_name}")
        pass


def resize_if_needed(model_name, model, token_nums):
    try:
        if model_name in MIS_MATCH_VOCAB_SIZE_MODELS:
            model.resize_token_embeddings(token_nums)
    except:
        print(f"Error resizing token embeddings for model {model_name}")
        pass


def average_checkpoints(output_dir: str, submission_dir: str) -> bool:
    """
    Average weights from all saved checkpoints in output_dir and write the
    result to submission_dir, replacing whatever the best-checkpoint callback
    already put there.

    Returns True  if averaging was performed (≥ 2 checkpoints found).
    Returns False if skipped (< 2 checkpoints).
    Must be called only from the main process.
    """
    import glob as _glob

    checkpoint_dirs = sorted(
        _glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )

    if len(checkpoint_dirs) < 2:
        print(
            f"[Checkpoint Averaging] Only {len(checkpoint_dirs)} checkpoint(s) found — "
            f"skipping averaging.",
            flush=True,
        )
        return False

    print(
        f"[Checkpoint Averaging] Averaging {len(checkpoint_dirs)} checkpoints: "
        + ", ".join(os.path.basename(d) for d in checkpoint_dirs),
        flush=True,
    )

    try:
        from safetensors.torch import (
            load_file as _load_st,
            save_file as _save_st,
        )
        HAS_ST = True
    except ImportError:
        HAS_ST = False

    is_lora = os.path.exists(
        os.path.join(checkpoint_dirs[0], "adapter_config.json")
    )
    n = len(checkpoint_dirs)
    avg_weights: dict = {}

    if is_lora:
        for ckpt_dir in checkpoint_dirs:
            st_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
            bin_path = os.path.join(ckpt_dir, "adapter_model.bin")
            if HAS_ST and os.path.exists(st_path):
                weights = _load_st(st_path, device="cpu")
            elif os.path.exists(bin_path):
                weights = torch.load(bin_path, map_location="cpu")
            else:
                print(
                    f"[Checkpoint Averaging] No adapter weights in {ckpt_dir} — skipping.",
                    flush=True,
                )
                continue
            for key, val in weights.items():
                if key not in avg_weights:
                    avg_weights[key] = val.float() / n
                else:
                    avg_weights[key] += val.float() / n
    else:
        # Full model — handle single-file and sharded checkpoints
        for ckpt_dir in checkpoint_dirs:
            shard_files: list = []
            if HAS_ST:
                shard_files = sorted(_glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
            if not shard_files:
                shard_files = sorted(_glob.glob(os.path.join(ckpt_dir, "*.bin")))

            for sf in shard_files:
                if HAS_ST and sf.endswith(".safetensors"):
                    shard = _load_st(sf, device="cpu")
                else:
                    shard = torch.load(sf, map_location="cpu")
                for key, val in shard.items():
                    if key not in avg_weights:
                        avg_weights[key] = val.float() / n
                    else:
                        avg_weights[key] += val.float() / n
                del shard

    if not avg_weights:
        print("[Checkpoint Averaging] No weights collected — skipping.", flush=True)
        return False

    # Cast back to bfloat16
    avg_weights = {k: v.to(torch.bfloat16) for k, v in avg_weights.items()}

    # Use last checkpoint as the template (configs, tokenizer files, etc.)
    last_ckpt = checkpoint_dirs[-1]
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    shutil.copytree(last_ckpt, submission_dir)

    if is_lora:
        if HAS_ST:
            _save_st(avg_weights, os.path.join(submission_dir, "adapter_model.safetensors"))
            bin_path = os.path.join(submission_dir, "adapter_model.bin")
            if os.path.exists(bin_path):
                os.remove(bin_path)
        else:
            torch.save(avg_weights, os.path.join(submission_dir, "adapter_model.bin"))
    else:
        # Remove old weight files and shard indices from the copied directory
        for old_f in (
            _glob.glob(os.path.join(submission_dir, "*.safetensors"))
            + _glob.glob(os.path.join(submission_dir, "*.bin"))
        ):
            os.remove(old_f)
        for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            idx_path = os.path.join(submission_dir, idx_name)
            if os.path.exists(idx_path):
                os.remove(idx_path)
        # Save as a single file
        if HAS_ST:
            _save_st(avg_weights, os.path.join(submission_dir, "model.safetensors"))
        else:
            torch.save(avg_weights, os.path.join(submission_dir, "pytorch_model.bin"))

    # Record what was done
    with open(os.path.join(submission_dir, "loss.txt"), "w") as f:
        f.write(f"averaged,{n}_checkpoints")

    print(f"[Checkpoint Averaging] Saved averaged model to {submission_dir}", flush=True)
    return True


def init_wandb(train_request: Dict):
    # set wandb_mode=offline; do not upload the data to wandb export WANDB_MODE=offline
    return True
    task_id = train_request["task_id"]
    expected_repo_name = train_request["expected_repo_name"]
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = train_request["wandb_log_dir"]
    os.environ["WANDB_RUN_ID"] = f"{task_id}_{expected_repo_name}"
    os.environ["WANDB_NAME"] = f"{task_id}_{expected_repo_name}"
    if is_main_process(LOCAL_RANK):
        os.makedirs(train_request["wandb_log_dir"], exist_ok=True)
    return True