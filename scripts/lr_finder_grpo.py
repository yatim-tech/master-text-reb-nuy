"""
lr_finder_grpo.py — GRPO-bespoke LR Finder using Reward Signal

Unlike the standard LR finder (lr_finder_les.py) which uses CE loss as a metric,
this finder uses *reward improvement* as the signal to select the optimal LR.

Why reward and not loss?
  - GRPO loss is not an absolute value; it depends on reward scale and clipping
  - Reward improvement is the actual downstream objective we care about
  - Allows the finder to stay true to the GRPO training objective

Algorithm:
  1. Save LoRA model state
  2. Compute baseline reward on a mini-batch (before any update)
  3. For each LR candidate (log-spaced):
       a. Restore model state
       b. Run N GRPO-lite steps (gen + reward + policy gradient)
       c. Re-generate on same prompts → measure reward_after
       d. Record reward_delta = reward_after - reward_baseline
  4. Select LR with highest reward_delta (early-stop on collapse)
  5. Restore original model state, return selected LR

DDP-safe: reward_delta is all-reduced across ranks after each candidate.
"""

import time
import logging
import numpy as np
import torch
import gc
import os
from typing import List, Callable, Optional
# NOTE: AdamW is intentionally NOT imported.
# It stores m1+m2 moment vectors per parameter → can OOM near-full GPUs.
# We use direct gradient update (param.data -= lr * grad) instead — same
# approach as lr_finder_les.py Fix 1.

logger = logging.getLogger("lr_finder_grpo")

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


# ─────────────────────────────────────────── DDP helpers ─────────────────────


def _is_main():
    return LOCAL_RANK == 0


def _dist_avg(value: float) -> float:
    """Average a scalar across all DDP ranks."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and WORLD_SIZE > 1:
            t = torch.tensor(value, dtype=torch.float64, device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            return t.item()
    except Exception:
        pass
    return value


def _barrier():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and WORLD_SIZE > 1:
            dist.barrier()
    except Exception:
        pass


# ──────────────────────────────────────── State helpers ──────────────────────


def _save_lora_state(model) -> dict:
    """Save only LoRA + lm_head weights (CPU clone, minimal VRAM)."""
    base = model.module if hasattr(model, "module") else model
    return {
        name: param.detach().cpu().clone()
        for name, param in base.named_parameters()
        if ("lora_" in name or "lm_head" in name) and param.requires_grad
    }


def _restore_lora_state(model, state: dict):
    """Restore LoRA + lm_head weights from a CPU state dict."""
    base = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for name, param in base.named_parameters():
            if name in state:
                param.copy_(state[name].to(param.device))


# ──────────────────────────────────────── Reward helpers ─────────────────────


def _generate_completions(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> List[str]:
    """
    Greedy generation of short completions for reward measurement.
    Short + greedy = fast, deterministic, low VRAM.
    """
    completions = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # greedy — fast & deterministic
                pad_token_id=tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            completions.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    model.train()
    return completions


def _compute_mean_reward(
    reward_funcs: List[Callable],
    completions: List[str],
    extra_data: Optional[List] = None,
) -> float:
    """
    Aggregate reward across all reward functions (equal weight).
    Handles both standard and extra_data-aware reward funcs.
    """
    import inspect
    total = 0.0
    count = 0
    for func in reward_funcs:
        try:
            sig = inspect.signature(func)
            if "extra_data" in sig.parameters and extra_data is not None:
                rewards = func(completions, extra_data=extra_data)
            else:
                rewards = func(completions)
            total += float(np.mean(rewards))
            count += 1
        except Exception as e:
            logger.warning(f"[LR Finder/GRPO] reward func error (skipped): {e}")
    return total / count if count > 0 else 0.0


# ──────────────────────────────────────── GRPO-lite step ─────────────────────


def _grpo_lite_steps(
    model,
    tokenizer,
    lr: float,
    prompts: List[str],
    reward_funcs: List[Callable],
    extra_data: Optional[List],
    n_steps: int = 3,
    max_gen_tokens: int = 64,
    num_generations: int = 2,
    beta: float = 0.04,
    device: str = "cuda",
):
    """
    Minimal GRPO update loop using direct gradient descent (no AdamW/SGD).

    Why no AdamW?
      AdamW stores m1 and m2 moment vectors for every trainable parameter.
      On near-full GPUs this alone can cause OOM (same reason as lr_finder_les.py).
      Direct gradient update (param -= lr * grad) requires zero extra VRAM.

    Steps per iteration:
      - Generate num_generations completions per prompt (short, greedy)
      - Compute rewards via reward_funcs
      - Compute advantage-weighted policy gradient loss
      - loss.backward() then apply param.data -= lr * param.grad directly
    """
    import inspect
    model.train()
    base = model.module if hasattr(model, "module") else model

    for _ in range(n_steps):
        all_prompts = []
        all_completions = []

        # Generate multiple completions per prompt
        for prompt in prompts:
            for _ in range(num_generations):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_gen_tokens,
                        do_sample=True,
                        temperature=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
                completion = tokenizer.decode(new_ids, skip_special_tokens=True)
                all_prompts.append(prompt)
                all_completions.append(completion)

        # Compute rewards for all completions
        raw_rewards = []
        for func in reward_funcs:
            try:
                sig = inspect.signature(func)
                if "extra_data" in sig.parameters and extra_data is not None:
                    expanded_extra = []
                    for ed in extra_data:
                        for _ in range(num_generations):
                            expanded_extra.append(ed)
                    r = func(all_completions, extra_data=expanded_extra)
                else:
                    r = func(all_completions)
                raw_rewards.append(r)
            except Exception:
                raw_rewards.append([0.0] * len(all_completions))

        # Average across reward functions
        rewards_tensor = torch.tensor(
            [float(np.mean([raw_rewards[f][i] for f in range(len(raw_rewards))]))
             for i in range(len(all_completions))],
            dtype=torch.float32,
            device=device,
        )

        # Compute advantages: normalize per prompt group
        advantages = rewards_tensor.clone()
        for pi in range(len(prompts)):
            start = pi * num_generations
            end = start + num_generations
            group = rewards_tensor[start:end]
            mean = group.mean()
            std = group.std().clamp(min=1e-4)
            advantages[start:end] = (group - mean) / std

        # Zero gradients manually (no optimizer object needed)
        for p in base.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Compute advantage-weighted policy gradient loss
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for i, (prompt, completion) in enumerate(zip(all_prompts, all_completions)):
            full_text = prompt + completion
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=768,
            ).to(device)
            prompt_len = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"].shape[1]
            labels = inputs["input_ids"].clone()
            labels[0, :prompt_len] = -100
            outputs = model(**inputs, labels=labels)
            adv = advantages[i].detach()
            loss_i = -adv * (-outputs.loss)
            total_loss = total_loss + loss_i / len(all_completions)

        total_loss.backward()

        # ── Direct gradient update — no optimizer state (VRAM-safe) ──────────
        # Clip first so large gradients don't produce out-of-range LR steps.
        torch.nn.utils.clip_grad_norm_(
            [p for p in base.parameters() if p.requires_grad], max_norm=1.0
        )
        with torch.no_grad():
            for p in base.parameters():
                if p.requires_grad and p.grad is not None:
                    p.data -= lr * p.grad  # pure SGD step, zero extra VRAM


# ═════════════════════════════════════ Main public API ════════════════════════


def find_grpo_lr(
    model,
    tokenizer,
    reward_funcs: List[Callable],
    dataset,
    start_lr: float = 5e-6,
    end_lr: float = 5e-5,
    num_candidates: int = 8,
    steps_per_candidate: int = 3,
    max_gen_tokens: int = 64,
    num_generations: int = 2,
    time_budget_minutes: float = 8.0,
    lora: bool = True,
    beta: float = 0.04,
    sample_size: int = 6,
) -> float:
    """
    GRPO-bespoke LR finder using reward improvement as the selection metric.

    Args:
        model:               The GRPO model (may be DDP-wrapped)
        tokenizer:           Tokenizer
        reward_funcs:        List of callable reward functions (same as GRPOTrainer)
        dataset:             Dataset with 'prompt' column (uses dev_ds, ~200 samples)
        start_lr:            Minimum LR to explore
        end_lr:              Maximum LR to explore
        num_candidates:      Number of LR candidates to try (default: 8)
        steps_per_candidate: GRPO-lite update steps per LR candidate
        max_gen_tokens:      Max new tokens during finder (keep short for speed)
        num_generations:     Completions per prompt during GRPO-lite step
        time_budget_minutes: Hard time limit for the entire finder
        lora:                If True, saves/restores only LoRA + lm_head params
        beta:                GRPO beta (unused in lite version, for future)
        sample_size:         Number of prompts to use per candidate evaluation

    Returns:
        float: Selected learning rate
    """
    device = f"cuda:{LOCAL_RANK}"
    time_budget_seconds = time_budget_minutes * 60
    start_time = time.time()

    if _is_main():
        logger.info(
            f"[LR Finder/GRPO] Starting — WORLD_SIZE={WORLD_SIZE}, "
            f"start_lr={start_lr:.2e}, end_lr={end_lr:.2e}, "
            f"candidates={num_candidates}, steps/candidate={steps_per_candidate}, "
            f"max_gen_tokens={max_gen_tokens}, time_budget={time_budget_minutes:.1f}min"
        )

    # ── 1. Save model state ──────────────────────────────────────────────────
    if lora:
        saved_state = _save_lora_state(model)
    else:
        base = model.module if hasattr(model, "module") else model
        saved_state = {
            name: param.detach().cpu().clone()
            for name, param in base.named_parameters()
            if param.requires_grad
        }

    # ── 2. Prepare mini-batch from dataset ──────────────────────────────────
    ds_list = dataset.to_list()
    sample_size = min(sample_size, len(ds_list))
    # Use a fixed seed slice for reproducibility across candidates
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(ds_list), size=sample_size, replace=False)
    sample = [ds_list[i] for i in indices]

    prompts = [row["prompt"] for row in sample]
    extra_data = [row.get("extra_data") for row in sample]
    has_extra = any(e is not None for e in extra_data)
    if not has_extra:
        extra_data = None

    if _is_main():
        logger.info(
            f"[LR Finder/GRPO] Using {sample_size} prompts for reward measurement. "
            f"has_extra_data={has_extra}"
        )

    # ── 3. Compute baseline reward ───────────────────────────────────────────
    baseline_completions = _generate_completions(
        model, tokenizer, prompts,
        max_new_tokens=max_gen_tokens, device=device
    )
    baseline_reward = _compute_mean_reward(reward_funcs, baseline_completions, extra_data)
    baseline_reward = _dist_avg(baseline_reward)

    if _is_main():
        logger.info(f"[LR Finder/GRPO] Baseline reward: {baseline_reward:.4f}")

    _barrier()

    # ── 4. Sweep over LR candidates ─────────────────────────────────────────
    lr_candidates = np.logspace(
        np.log10(start_lr), np.log10(end_lr), num=num_candidates
    )
    reward_deltas = []
    best_lr = float(lr_candidates[num_candidates // 2])  # fallback: middle

    collapse_threshold = max(0.0, baseline_reward * 0.3)  # reward dropped 70%

    for i, candidate_lr in enumerate(lr_candidates):
        if time.time() - start_time > time_budget_seconds * 0.85:
            if _is_main():
                logger.info(
                    f"[LR Finder/GRPO] Time budget reached at candidate {i+1}/{num_candidates}"
                )
            break

        # Restore original state before each candidate
        _restore_lora_state(model, saved_state)
        gc.collect()
        torch.cuda.empty_cache()
        _barrier()

        if _is_main():
            logger.info(
                f"[LR Finder/GRPO] Candidate {i+1}/{num_candidates}, LR: {candidate_lr:.2e}"
            )

        # Run GRPO-lite update steps (direct gradient, no AdamW)
        try:
            _grpo_lite_steps(
                model=model,
                tokenizer=tokenizer,
                lr=float(candidate_lr),
                prompts=prompts,
                reward_funcs=reward_funcs,
                extra_data=extra_data,
                n_steps=steps_per_candidate,
                max_gen_tokens=max_gen_tokens,
                num_generations=num_generations,
                beta=beta,
                device=device,
            )
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if _is_main():
                    logger.warning(
                        f"[LR Finder/GRPO] OOM at LR={candidate_lr:.2e}, skipping."
                    )
                torch.cuda.empty_cache()
                reward_deltas.append(-9999.0)
                continue
            else:
                raise

        # Measure reward after update
        after_completions = _generate_completions(
            model, tokenizer, prompts,
            max_new_tokens=max_gen_tokens, device=device
        )
        after_reward = _compute_mean_reward(reward_funcs, after_completions, extra_data)
        after_reward = _dist_avg(after_reward)

        delta = after_reward - baseline_reward
        reward_deltas.append(delta)

        if _is_main():
            logger.info(
                f"[LR Finder/GRPO] LR={candidate_lr:.2e} → "
                f"reward_before={baseline_reward:.4f}, "
                f"reward_after={after_reward:.4f}, "
                f"delta={delta:+.4f}"
            )

        # Early stop: catastrophic reward collapse
        if after_reward < collapse_threshold and baseline_reward > 0:
            if _is_main():
                logger.warning(
                    f"[LR Finder/GRPO] Reward collapsed at LR={candidate_lr:.2e} "
                    f"({after_reward:.4f} < {collapse_threshold:.4f}). Stopping sweep."
                )
            break

    # ── 5. Select best LR ────────────────────────────────────────────────────
    evaluated_lrs = lr_candidates[:len(reward_deltas)]

    if len(reward_deltas) == 0 or all(d <= -9999 for d in reward_deltas):
        if _is_main():
            logger.warning(
                "[LR Finder/GRPO] No valid reward deltas. Falling back to start_lr."
            )
        best_lr = float(start_lr)
    else:
        valid_mask = [d > -9999 for d in reward_deltas]
        valid_deltas = [d for d, m in zip(reward_deltas, valid_mask) if m]
        valid_lrs = [lr for lr, m in zip(evaluated_lrs, valid_mask) if m]

        best_idx = int(np.argmax(valid_deltas))
        best_lr = float(valid_lrs[best_idx])

        if _is_main():
            logger.info(
                f"[LR Finder/GRPO] Best LR: {best_lr:.2e} "
                f"(reward_delta={valid_deltas[best_idx]:+.4f})"
            )

    # ── 6. Restore original model state ─────────────────────────────────────
    _restore_lora_state(model, saved_state)
    del saved_state
    gc.collect()
    torch.cuda.empty_cache()
    _barrier()

    elapsed = time.time() - start_time
    if _is_main():
        logger.info(
            f"[LR Finder/GRPO] Done. Selected LR = {best_lr:.2e} "
            f"(elapsed {elapsed:.1f}s)"
        )

    return best_lr
