"""
lr_finder_grpo.py — GRPO-bespoke LR Finder using Reward-Weighted Log-Prob Signal

Metric: reward-weighted log-probability improvement.

Why NOT reward re-generation?
  After only 3 gradient steps at LR=2e-6..32e-6, the weight delta is ~6e-6.
  This is too small to flip any argmax token under greedy decoding, so
  reward stays identical (all delta=0) regardless of LR.

Why log-prob improvement?
  Even a 6e-6 weight change measurably shifts log P(completion|prompt).
  Weighted by reward, this directly measures what GRPO optimises:
  "Does this LR cause the model to assign higher probability to good completions?"

  signal(lr) = mean( reward_i * (log_p_after_i - log_p_before_i) )

  Higher signal → this LR is better at reinforcing high-reward completions.

DDP-safe: signal is all_reduce'd across ranks after each candidate.
AdamW-free: uses direct gradient update (param -= lr * grad), zero extra VRAM.
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


def _save_full_state(model) -> dict:
    """Save all trainable weights (CPU clone)."""
    base = model.module if hasattr(model, "module") else model
    return {
        name: param.detach().cpu().clone()
        for name, param in base.named_parameters()
        if param.requires_grad
    }


def _restore_state(model, state: dict):
    """Restore saved weights from a CPU state dict."""
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
            new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            completions.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    model.train()
    return completions


def _compute_rewards(
    reward_funcs: List[Callable],
    completions: List[str],
    extra_data: Optional[List] = None,
) -> List[float]:
    """
    Compute per-completion reward (averaged across all reward functions).
    Returns a list of float rewards, one per completion.
    """
    import inspect
    all_rewards = []
    for func in reward_funcs:
        try:
            sig = inspect.signature(func)
            if "extra_data" in sig.parameters and extra_data is not None:
                r = func(completions, extra_data=extra_data)
            else:
                r = func(completions)
            all_rewards.append([float(x) for x in r])
        except Exception as e:
            logger.warning(f"[LR Finder/GRPO] reward func error (skipped): {e}")
            all_rewards.append([0.0] * len(completions))

    if not all_rewards:
        return [0.0] * len(completions)
    # Average across reward functions per completion
    return [float(np.mean([all_rewards[f][i] for f in range(len(all_rewards))]))
            for i in range(len(completions))]


def _compute_logprobs(
    model,
    tokenizer,
    prompts: List[str],
    completions: List[str],
    device: str = "cuda",
) -> List[float]:
    """
    Compute mean log-probability of each completion given its prompt.
    Only completion tokens are scored (prompt tokens are masked).
    This is sensitive to even tiny weight changes — unlike greedy output.
    """
    model.eval()
    logprobs = []
    with torch.no_grad():
        for prompt, completion in zip(prompts, completions):
            if not completion.strip():
                logprobs.append(0.0)
                continue

            full_text = prompt + completion
            enc_full = tokenizer(
                full_text, return_tensors="pt",
                truncation=True, max_length=768,
            ).to(device)
            enc_prompt = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=512,
            ).to(device)

            prompt_len = enc_prompt["input_ids"].shape[1]
            input_ids = enc_full["input_ids"]
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100  # mask prompt

            outputs = model(**enc_full, labels=labels)
            # outputs.loss = mean NLL of completion tokens (scalar)
            logprobs.append(-float(outputs.loss))  # higher is better

    model.train()
    return logprobs


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
    device: str = "cuda",
):
    """
    Minimal GRPO update loop using direct gradient descent (no AdamW/SGD).

    Why no AdamW?
      AdamW stores m1 and m2 moment vectors for every trainable parameter.
      On near-full GPUs this alone can cause OOM (same reason as lr_finder_les.py Fix 1).
      Direct gradient update (param -= lr * grad) requires zero extra VRAM.
    """
    import inspect
    model.train()
    base = model.module if hasattr(model, "module") else model

    for _ in range(n_steps):
        all_prompts = []
        all_completions = []

        # Generate completions (stochastic — diverse signal for advantage)
        for prompt in prompts:
            for _ in range(num_generations):
                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=512,
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

        # Compute rewards
        raw_rewards = []
        for func in reward_funcs:
            try:
                sig = inspect.signature(func)
                if "extra_data" in sig.parameters and extra_data is not None:
                    expanded_extra = [ed for ed in extra_data for _ in range(num_generations)]
                    r = func(all_completions, extra_data=expanded_extra)
                else:
                    r = func(all_completions)
                raw_rewards.append(r)
            except Exception:
                raw_rewards.append([0.0] * len(all_completions))

        rewards_tensor = torch.tensor(
            [float(np.mean([raw_rewards[f][i] for f in range(len(raw_rewards))]))
             for i in range(len(all_completions))],
            dtype=torch.float32, device=device,
        )

        # Advantages: normalize per-prompt group
        advantages = rewards_tensor.clone()
        for pi in range(len(prompts)):
            start = pi * num_generations
            end = start + num_generations
            group = rewards_tensor[start:end]
            mean = group.mean()
            std = group.std().clamp(min=1e-4)
            advantages[start:end] = (group - mean) / std

        # Zero gradients
        for p in base.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Advantage-weighted policy gradient loss
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for i, (prompt, completion) in enumerate(zip(all_prompts, all_completions)):
            full_text = prompt + completion
            inputs = tokenizer(
                full_text, return_tensors="pt",
                truncation=True, max_length=768,
            ).to(device)
            prompt_len = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=512,
            )["input_ids"].shape[1]
            labels = inputs["input_ids"].clone()
            labels[0, :prompt_len] = -100
            outputs = model(**inputs, labels=labels)
            adv = advantages[i].detach()
            loss_i = -adv * (-outputs.loss)
            total_loss = total_loss + loss_i / len(all_completions)

        total_loss.backward()

        # Direct gradient update — no optimizer state (VRAM-safe)
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
    sample_size: int = 6,
) -> float:
    """
    GRPO-bespoke LR finder using reward-weighted log-prob improvement as metric.

    Signal per LR candidate:
        signal = mean( reward_i * (log_p_after_i - log_p_before_i) )

    This measures how much each LR causes the model to assign higher
    probability to high-reward completions — exactly what GRPO optimises.
    It is sensitive to weight changes as small as 1e-6, unlike reward
    re-generation which requires a token-level argmax flip.

    Args:
        model:               The GRPO model (optionally DDP-wrapped)
        tokenizer:           Tokenizer
        reward_funcs:        List of callable reward functions
        dataset:             Dataset with 'prompt' column (uses dev_ds)
        start_lr / end_lr:   LR search range
        num_candidates:      Number of LR candidates (log-spaced)
        steps_per_candidate: GRPO-lite gradient steps per candidate
        max_gen_tokens:      Max new tokens for baseline + lite generation
        num_generations:     Completions per prompt in GRPO-lite step
        time_budget_minutes: Hard time ceiling
        lora:                If True saves/restores only LoRA + lm_head params
        sample_size:         Number of prompts to sample from dataset

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
    saved_state = _save_lora_state(model) if lora else _save_full_state(model)

    # ── 2. Prepare fixed mini-batch ──────────────────────────────────────────
    ds_list = dataset.to_list()
    sample_size = min(sample_size, len(ds_list))
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
            f"[LR Finder/GRPO] {sample_size} prompts, has_extra_data={has_extra}"
        )

    # ── 3. Baseline: generate completions + measure rewards + log-probs ──────
    baseline_completions = _generate_completions(
        model, tokenizer, prompts, max_new_tokens=max_gen_tokens, device=device
    )
    baseline_rewards = _compute_rewards(reward_funcs, baseline_completions, extra_data)
    baseline_logprobs = _compute_logprobs(model, tokenizer, prompts, baseline_completions, device)

    # Average baseline reward just for logging
    baseline_reward_mean = _dist_avg(float(np.mean(baseline_rewards)))
    if _is_main():
        logger.info(
            f"[LR Finder/GRPO] Baseline — reward: {baseline_reward_mean:.4f}, "
            f"mean_logprob: {float(np.mean(baseline_logprobs)):.4f}"
        )

    _barrier()

    # ── 4. Sweep ──────────────────────────────────────────────────────────────
    lr_candidates = np.logspace(
        np.log10(start_lr), np.log10(end_lr), num=num_candidates
    )
    signals = []  # reward_weighted log-prob improvement per candidate

    for i, candidate_lr in enumerate(lr_candidates):
        if time.time() - start_time > time_budget_seconds * 0.85:
            if _is_main():
                logger.info(f"[LR Finder/GRPO] Time budget reached at candidate {i+1}")
            break

        _restore_state(model, saved_state)
        gc.collect()
        torch.cuda.empty_cache()
        _barrier()

        if _is_main():
            logger.info(f"[LR Finder/GRPO] Candidate {i+1}/{num_candidates}, LR: {candidate_lr:.2e}")

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
                device=device,
            )
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if _is_main():
                    logger.warning(f"[LR Finder/GRPO] OOM at LR={candidate_lr:.2e}, skipping.")
                torch.cuda.empty_cache()
                signals.append(-9999.0)
                _barrier()
                continue
            else:
                raise

        # Measure log-prob of same baseline completions under updated model
        after_logprobs = _compute_logprobs(
            model, tokenizer, prompts, baseline_completions, device
        )

        # Reward-weighted log-prob improvement
        # signal = sum(reward_i * delta_logprob_i) — positive means high-reward
        # completions became MORE likely (good), negative means less likely (bad).
        delta_logprobs = [a - b for a, b in zip(after_logprobs, baseline_logprobs)]
        signal = float(np.mean([
            r * d for r, d in zip(baseline_rewards, delta_logprobs)
        ]))
        signal = _dist_avg(signal)  # consensus across ranks
        signals.append(signal)

        if _is_main():
            logger.info(
                f"[LR Finder/GRPO] LR={candidate_lr:.2e} → "
                f"signal={signal:+.6f} "
                f"(mean_Δlogp={float(np.mean(delta_logprobs)):+.4f})"
            )

        # Early stop: model collapsed (all log-probs crashed)
        mean_after_lp = float(np.mean(after_logprobs))
        if mean_after_lp < float(np.mean(baseline_logprobs)) - 2.0:
            if _is_main():
                logger.warning(
                    f"[LR Finder/GRPO] Log-prob collapse at LR={candidate_lr:.2e}, stopping."
                )
            break

    # ── 5. Select best LR ────────────────────────────────────────────────────
    evaluated_lrs = lr_candidates[:len(signals)]

    if len(signals) == 0 or all(s <= -9999 for s in signals):
        best_lr = float(start_lr)
        if _is_main():
            logger.warning("[LR Finder/GRPO] No valid signals. Falling back to start_lr.")
    else:
        valid_mask = [s > -9999 for s in signals]
        valid_signals = [s for s, m in zip(signals, valid_mask) if m]
        valid_lrs = [lr for lr, m in zip(evaluated_lrs, valid_mask) if m]
        best_idx = int(np.argmax(valid_signals))
        best_lr = float(valid_lrs[best_idx])
        if _is_main():
            logger.info(
                f"[LR Finder/GRPO] Best LR: {best_lr:.2e} "
                f"(signal={valid_signals[best_idx]:+.6f})"
            )

    # ── 6. Restore original model state ─────────────────────────────────────
    _restore_state(model, saved_state)
    del saved_state
    gc.collect()
    torch.cuda.empty_cache()
    _barrier()

    elapsed = time.time() - start_time
    if _is_main():
        logger.info(
            f"[LR Finder/GRPO] Done. Selected LR = {best_lr:.2e} (elapsed {elapsed:.1f}s)"
        )

    return best_lr
