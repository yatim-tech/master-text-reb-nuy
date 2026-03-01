import time
import logging
import numpy as np
import torch
import gc
import os
from torch.optim import AdamW
from dataclasses import dataclass
from config_pair import RATIO_BOST

logger = logging.getLogger(__name__)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


def _is_ddp_model(model) -> bool:
    """Check if the model is wrapped in DistributedDataParallel."""
    return hasattr(model, "module") and isinstance(
        model.module, torch.nn.Module
    )


def _get_base_model(model):
    """Unwrap DDP if needed to get the actual model."""
    if _is_ddp_model(model):
        return model.module
    return model


def _dist_avg_scalar(value: float) -> float:
    """Average a scalar across all DDP ranks (no-op if not distributed)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and WORLD_SIZE > 1:
            t = torch.tensor(value, dtype=torch.float64, device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            return t.item()
    except Exception:
        pass
    return value


def _dist_barrier():
    """Barrier across all ranks (no-op if not distributed)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and WORLD_SIZE > 1:
            dist.barrier()
    except Exception:
        pass


def analyze_loss_landscape(model, batch, lr, trainer, num_points=5, radius=0.1, lora=True):
    """
    Quick analysis of the loss landscape around the given learning rate.
    DDP-safe: works on the unwrapped base model.

    Returns:
        Dictionary with landscape metrics
    """
    base_model = _get_base_model(model)

    # Save original parameters (always use base model's named_parameters)
    if lora:
        original_params = {
            name: param.detach().clone()
            for name, param in base_model.named_parameters()
            if "lora_" in name or "lm_head" in name
        }
    else:
        original_params = {
            name: param.detach().clone()
            for name, param in base_model.named_parameters()
        }

    # Get initial loss and gradient
    loss = trainer.compute_loss(model, batch)
    loss.backward()

    # Calculate gradient norm for relevant parameters
    grad_params = []
    for name, param in base_model.named_parameters():
        if param.grad is not None:
            if lora:
                if "lora_" in name or "lm_head" in name:
                    grad_params.append(torch.norm(param.grad))
            else:
                grad_params.append(torch.norm(param.grad))

    grad_norm = torch.norm(torch.stack(grad_params)) if grad_params else torch.tensor(0.0)

    # Evaluate loss at different learning rates around the current one
    lr_factors = np.linspace(1 - radius, 1 + radius, num_points)
    losses = []

    for factor in lr_factors:
        current_lr = lr * factor

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if param.grad is not None:
                    if lora:
                        if "lora_" in name or "lm_head" in name:
                            param.data = original_params[name] - current_lr * param.grad
                    else:
                        param.data = original_params[name] - current_lr * param.grad

        loss_val = float(trainer.compute_loss(model, batch))
        # Average loss across ranks for a consistent landscape
        loss_val = _dist_avg_scalar(loss_val)
        losses.append(loss_val)

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

    losses = np.array(losses)
    min_idx = np.argmin(losses)

    smoothness = np.mean(np.abs(np.diff(losses, 2))) if len(losses) > 2 else float("inf")
    convexity = np.mean(np.diff(losses, 2)) if len(losses) > 2 else 0
    is_centered = min_idx in [num_points // 2 - 1, num_points // 2, num_points // 2 + 1]

    metrics = {
        "smoothness": smoothness,
        "convexity": convexity,
        "centered": is_centered,
        "grad_norm": float(grad_norm),
        "min_loss": float(np.min(losses)),
        "min_lr_factor": lr_factors[min_idx],
    }

    del original_params
    torch.cuda.empty_cache()

    return metrics


def find_lr_and_continue(
    trainer,
    start_lr: float = 1e-6,
    end_lr: float = 1e-3,
    time_budget_minutes: float = 5.0,
    warmup_fraction: float = 0.1,
    num_candidates: int = 2,
    preserve_state: bool = True,
    lora: bool = True,
    use_gradient_checkpointing: bool = True,
    finder_batch_size: int = None,
):
    """
    Multi-GPU-safe LR finder based on Leslie Smith's approach.

    Multi-GPU strategy:
    - All ranks run the LR finder in PARALLEL (no idle ranks).
    - Each rank explores LRs on its own local data shard.
    - At the end, the chosen LR is averaged across all ranks via all_reduce.
    - A dist.barrier() is inserted at critical points to keep ranks in sync.

    The key insight: since trainer.compute_loss() already triggers
    gradient all-reduce in DDP, all ranks must call it at the same step and
    with the same number of steps, so they stay in sync naturally.

    Args:
        trainer: HuggingFace Trainer instance
        start_lr: Minimum LR to test
        end_lr: Maximum LR to test
        time_budget_minutes: Max time for the LR finder phase
        warmup_fraction: Unused, kept for API compatibility
        num_candidates: Unused, kept for API compatibility
        preserve_state: Whether to save and restore model state after finder
        lora: If True, only track LoRA and lm_head parameters

    Returns:
        Tuple of (selected_lr, best_state_dict, None, num_evaluations)
    """
    model = trainer.model
    base_model = _get_base_model(model)

    time_budget_seconds = time_budget_minutes * 60
    start_time = time.time()

    rank = LOCAL_RANK
    is_main_rank = (rank == 0)

    # ------------------------------------------------------------------ #
    # 0. Temporarily halve the batch size for the finder sweep             #
    #    so activations + AdamW moment vectors don't OOM.                  #
    #    The real training batch size is restored at the end of this func. #
    # ------------------------------------------------------------------ #
    _original_batch_size = trainer.args.per_device_train_batch_size
    if finder_batch_size is None:
        # Only reduce batch size on multi-GPU — NCCL collective ops hold activation
        # tensors across ranks simultaneously, doubling effective VRAM pressure.
        # Single-GPU has no such overhead so we keep the original batch size.
        if WORLD_SIZE > 1:
            finder_batch_size = max(1, _original_batch_size // 2)
        else:
            finder_batch_size = _original_batch_size
    if finder_batch_size != _original_batch_size:
        trainer.args.per_device_train_batch_size = finder_batch_size
        if is_main_rank:
            logger.info(
                f"[LR Finder] Using finder_batch_size={finder_batch_size} "
                f"(real batch_size={_original_batch_size}, world_size={WORLD_SIZE})"
            )

    # Enable gradient checkpointing to reduce activation memory during sweep
    if use_gradient_checkpointing:
        try:
            base_model.gradient_checkpointing_enable()
            if is_main_rank:
                logger.info("[LR Finder] Gradient checkpointing enabled inside finder")
        except Exception as _gc_err:
            if is_main_rank:
                logger.warning(f"[LR Finder] Could not enable gradient checkpointing: {_gc_err}")

    if is_main_rank:
        logger.info(
            f"[LR Finder] Starting — WORLD_SIZE={WORLD_SIZE}, "
            f"start_lr={start_lr:.2e}, end_lr={end_lr:.2e}, "
            f"time_budget={time_budget_minutes:.1f}min, lora={lora}"
        )

    # ------------------------------------------------------------------ #
    # 1. Save model state (always from base model to avoid DDP naming)     #
    # ------------------------------------------------------------------ #
    if lora:
        initial_state = {
            name: param.detach().cpu().clone()
            for name, param in base_model.named_parameters()
            if ("lora_" in name or "lm_head" in name) and param.requires_grad
        }
    else:
        initial_state = {
            name: param.detach().cpu().clone()
            for name, param in base_model.named_parameters()
            if param.requires_grad
        }

    best_state_dict = None

    # ------------------------------------------------------------------ #
    # 2. Measure time-per-step to budget evaluations                      #
    # ------------------------------------------------------------------ #
    test_steps = 3
    dataloader_iter = iter(trainer.get_train_dataloader())
    batch = next(dataloader_iter)
    batch = trainer._prepare_inputs(batch)

    model.train()

    # Fix 1: Use no_grad for timing to avoid allocating AdamW optimizer states
    # on GPU (moment vectors alone can exhaust remaining VRAM on large models).
    # Forward-pass time is proportional to full step time, which is sufficient
    # to estimate the per-step budget.
    test_start_time = time.time()
    for _ in range(test_steps):
        with torch.no_grad():
            _ = trainer.compute_loss(model, batch)

    time_per_step = (time.time() - test_start_time) / test_steps
    if is_main_rank:
        logger.info(f"[LR Finder] Time per step: {time_per_step:.4f}s")

    main_time_budget = time_budget_seconds * 0.8
    max_evaluations = int(main_time_budget / (time_per_step * 5))
    max_evaluations = min(max(10, max_evaluations), 30)

    if is_main_rank:
        logger.info(f"[LR Finder] Planning up to {max_evaluations} evaluations")

    # ------------------------------------------------------------------ #
    # 3. Restore initial parameters before the sweep starts               #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if lora and ("lora_" in name or "lm_head" in name) and name in initial_state:
                param.copy_(initial_state[name].to(param.device))
            elif not lora and name in initial_state:
                param.copy_(initial_state[name].to(param.device))

    # Fix 3: Aggressively free any cached allocations before the sweep.
    # On near-full GPUs the difference between "fits" and OOM is a few MiB.
    gc.collect()
    torch.cuda.empty_cache()

    # Sync all ranks before the sweep
    _dist_barrier()

    # ------------------------------------------------------------------ #
    # 4. LR range sweep — all ranks run simultaneously                    #
    # ------------------------------------------------------------------ #
    lr_multiplier = np.power(end_lr / start_lr, 1 / max_evaluations)

    lrs = []
    losses = []
    smoothed_losses = []

    best_loss = float("inf")
    min_loss_so_far = float("inf")
    divergence_counter = 0
    steps_per_evaluation = 5
    current_lr = start_lr
    evaluation_count = 0

    while evaluation_count < max_evaluations and time.time() - start_time < main_time_budget:
        evaluation_count += 1

        if is_main_rank:
            logger.info(
                f"[LR Finder] Eval {evaluation_count}/{max_evaluations}, LR: {current_lr:.2e}"
            )

        optimizer = AdamW(model.parameters(), lr=current_lr)
        batch_losses = []

        for step in range(steps_per_evaluation):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(trainer.get_train_dataloader())
                batch = next(dataloader_iter)

            batch = trainer._prepare_inputs(batch)
            optimizer.zero_grad()
            loss = trainer.compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss))

        avg_loss = np.mean(batch_losses)

        # Average the loss across all ranks so every rank agrees on the
        # divergence check — critical for keeping all ranks in sync.
        avg_loss = _dist_avg_scalar(avg_loss)

        lrs.append(current_lr)
        losses.append(avg_loss)

        if len(losses) == 1:
            smoothed_losses.append(avg_loss)
        else:
            smoothed_losses.append(0.7 * smoothed_losses[-1] + 0.3 * avg_loss)

        if is_main_rank:
            logger.info(
                f"[LR Finder] LR: {current_lr:.2e}, Loss: {avg_loss:.4f}, "
                f"Smoothed: {smoothed_losses[-1]:.4f}"
            )

        if preserve_state and avg_loss < best_loss:
            if best_state_dict is not None:
                del best_state_dict
                torch.cuda.empty_cache()
            best_loss = avg_loss
            best_state_dict = {
                name: param.detach().cpu().clone()
                for name, param in base_model.named_parameters()
                if param.requires_grad
            }

        if avg_loss < min_loss_so_far:
            min_loss_so_far = avg_loss
            divergence_counter = 0
        else:
            if avg_loss > min_loss_so_far * 1.10:
                divergence_counter += 1
            else:
                divergence_counter = 0

        if avg_loss > 6.0 * min_loss_so_far or np.isnan(avg_loss) or np.isinf(avg_loss):
            if is_main_rank:
                logger.info(
                    f"[LR Finder] Early stop — significant divergence at LR: {current_lr:.2e}"
                )
            # Barrier so all ranks break at the same eval
            _dist_barrier()
            break

        if divergence_counter >= 8:
            if is_main_rank:
                logger.info(
                    f"[LR Finder] Early stop — consistent loss increase at LR: {current_lr:.2e}"
                )
            _dist_barrier()
            break

        current_lr *= lr_multiplier

    # ------------------------------------------------------------------ #
    # 5. Select optimal LR from the sweep                                 #
    # ------------------------------------------------------------------ #
    if len(lrs) < 3:
        if is_main_rank:
            logger.warning("[LR Finder] Too few evaluations — using default LR")
        final_lr = 8e-5
    else:
        smoothed_losses_arr = np.array(smoothed_losses)
        lrs_arr = np.array(lrs)

        gradients = np.gradient(smoothed_losses_arr) / np.gradient(np.log10(lrs_arr))

        window_size = min(3, len(gradients) // 4)
        windowed_gradients = [
            np.mean(gradients[max(0, i - window_size): min(len(gradients), i + window_size + 1)])
            for i in range(len(gradients))
        ]
        steepest_idx = np.argmin(windowed_gradients)
        min_loss_idx = np.argmin(smoothed_losses_arr)

        if len(smoothed_losses_arr) > 5:
            loss_end_to_min_ratio = smoothed_losses_arr[-1] / np.min(smoothed_losses_arr)
            loss_exploration_quality = min(1.0, loss_end_to_min_ratio / 10.0)
        else:
            loss_exploration_quality = 0.5

        # Fix 4: Inflection-point-aware LR selection for better L2 loss.
        # The inflection point is where the 2nd derivative of the smoothed loss
        # turns positive — i.e., loss stops curving down and starts curving up.
        # This is the boundary of the convex bowl; we must stay left of it.
        # np.diff(arr, 2) produces len-2 array, so index offsets by +2.
        second_deriv = np.diff(smoothed_losses_arr, 2)
        inflection_candidates = np.where(second_deriv > 0)[0]
        if len(inflection_candidates) > 0:
            # +2 to account for the 2-step index shift from np.diff(n=2)
            inflection_idx = int(inflection_candidates[0]) + 2
        else:
            inflection_idx = len(lrs_arr)  # no inflection found — use full range

        if is_main_rank:
            logger.info(
                f"[LR Finder] Inflection point at idx={inflection_idx} "
                f"(LR≈{lrs_arr[min(inflection_idx, len(lrs_arr)-1)]:.2e})"
            )

        if steepest_idx > 1:
            if loss_exploration_quality > 0.8:
                divider = 1.3
            else:
                divider = 1.8

            # Stay safely on the descending side of the inflection point
            safe_idx = min(steepest_idx, max(0, inflection_idx - 2))
            candidate_lr = lrs_arr[safe_idx] / divider
            min_bound = lrs_arr[min_loss_idx] / 2.0
            final_lr = max(candidate_lr, min_bound)

            if is_main_rank:
                logger.info(
                    f"[LR Finder] Steepest descent at LR: {lrs_arr[steepest_idx]:.2e}, "
                    f"safe_idx={safe_idx}, "
                    f"min loss at LR: {lrs_arr[min_loss_idx]:.2e}, "
                    f"selected: {final_lr:.2e} (divider={divider:.1f})"
                )
        else:
            final_lr = lrs_arr[min_loss_idx] / 2.0
            if is_main_rank:
                logger.info(f"[LR Finder] Limited data — selected LR based on min loss: {final_lr:.2e}")

    # ------------------------------------------------------------------ #
    # 6. Optional landscape validation (rank 0 only, then broadcast)      #
    # ------------------------------------------------------------------ #
    remaining_time = time_budget_seconds - (time.time() - start_time)
    if remaining_time > 60:
        # Only rank 0 does landscape analysis (read-only with respect to loss)
        # We still call compute_loss which needs all ranks in sync.
        try:
            batch = next(dataloader_iter)
            batch = trainer._prepare_inputs(batch)

            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    if lora and ("lora_" in name or "lm_head" in name) and name in initial_state:
                        param.copy_(initial_state[name].to(param.device))
                    elif not lora and name in initial_state:
                        param.copy_(initial_state[name].to(param.device))

            landscape = analyze_loss_landscape(model, batch, final_lr, trainer, lora=lora)

            if not landscape["centered"]:
                adjustment = max(0.6, min(1.5, landscape["min_lr_factor"]))
                if is_main_rank:
                    logger.info(f"[LR Finder] Landscape adjustment factor: {adjustment:.2f}")
                final_lr *= adjustment

            model_size_based_min = 5e-6
            try:
                if hasattr(base_model, "config") and hasattr(base_model.config, "hidden_size"):
                    if base_model.config.hidden_size > 2048:
                        model_size_based_min = 1e-5
            except Exception:
                pass

            if final_lr < model_size_based_min:
                if is_main_rank:
                    logger.info(
                        f"[LR Finder] Clamping LR to minimum: {model_size_based_min:.2e}"
                    )
                final_lr = model_size_based_min

            if is_main_rank:
                logger.info(
                    f"[LR Finder] Landscape: smoothness={landscape['smoothness']:.4f}, "
                    f"convexity={landscape['convexity']:.4f}, centered={landscape['centered']}"
                )
        except Exception as e:
            if is_main_rank:
                logger.warning(f"[LR Finder] Landscape analysis skipped: {e}")

    # ------------------------------------------------------------------ #
    # 7. Apply small boost and ALL-REDUCE across ranks                    #
    # ------------------------------------------------------------------ #
    final_lr *= RATIO_BOST
    if is_main_rank:
        logger.info(f"[LR Finder] Applied 15% boost: {final_lr:.2e}")

    # Average the final LR across all ranks — each rank did independent search
    # on its own data shard, so we take the consensus value.
    final_lr = _dist_avg_scalar(final_lr)

    if is_main_rank:
        logger.info(f"[LR Finder] Final LR after all-reduce: {final_lr:.2e}")

    # ------------------------------------------------------------------ #
    # 8. Restore original model parameters                                #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if lora and ("lora_" in name or "lm_head" in name) and name in initial_state:
                param.copy_(initial_state[name].to(param.device))
            elif not lora and name in initial_state:
                param.copy_(initial_state[name].to(param.device))

    # Final barrier — all ranks must complete the restore before training starts
    _dist_barrier()

    del initial_state
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------ #
    # 9. Restore the real training batch size                              #
    # ------------------------------------------------------------------ #
    trainer.args.per_device_train_batch_size = _original_batch_size
    if is_main_rank:
        logger.info(
            f"[LR Finder] Restored per_device_train_batch_size={_original_batch_size}"
        )

    if is_main_rank:
        logger.info(f"[LR Finder] Done. Selected LR = {final_lr:.2e}")

    return final_lr, best_state_dict, None, evaluation_count
