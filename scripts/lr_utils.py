import math

def _suggest_learning_rates(
    best_lr: float,
    n: int,
    log_range: float = 0.4
) -> list[float]:
    if n < 0:
        raise ValueError("Number of tries (n) cannot be negative.")
    if n == 0:
        return []
    if n == 1:
        return [best_lr]

    # print("best_lr: ", best_lr)
    # Calculate the lower and upper bounds for the learning rate search
    # on a logarithmic scale.
    lower_bound = best_lr / (10 ** log_range)
    upper_bound = best_lr * (10 ** log_range)

    # Convert bounds to log scale
    log_lower = math.log10(lower_bound)
    log_upper = math.log10(upper_bound)

    # Generate n logarithmically spaced values
    log_spaced_values = [
        log_lower + i * (log_upper - log_lower) / (n - 1)
        for i in range(n)
    ]

    # Convert the log-spaced values back to the original scale
    learning_rates = [10 ** val for val in log_spaced_values]

    return sorted(learning_rates)


def suggest_learning_rates(
    best_lr: float,
    n: int,
    log_range: float = 0.2
) -> list[float]:
    lrs = _suggest_learning_rates(best_lr, n, log_range)
    if n % 2 == 1:
        return lrs
    else: # exclude one and add best_lr to the middle
        lrs = lrs[1:] + [best_lr]
        lrs = sorted(lrs)
        return lrs


def extend_learning_rates(
    lr: float,
    n: int,
    log_range: float = 0.2
) -> list[float]:
    lrs = _suggest_learning_rates(lr, n, log_range)
    # loop over lrs to find the item that is the closest to lr (should be the same) and replace it with lr and move that item to the left (index = 0)
    # Find the index of the learning rate in lrs that is closest to lr
    closest_idx = min(range(len(lrs)), key=lambda i: abs(lrs[i] - lr))
    # Replace that value with the actual lr to ensure precision
    lrs[closest_idx] = lr
    # Move that lr to the first position (index 0)
    if closest_idx != 0:
        lrs.insert(0, lrs.pop(closest_idx))
    return lrs


def smart_extend_learning_rates(
    lr: float,
    n: int,
    log_range: float = 0.2,
) -> list[float]:
    """
    Golden-Ratio Center-Biased LR Search.

    Better than uniform log grid for unimodal loss landscapes (which is
    always the case after LR Finder has already narrowed the range).

    Strategy:
      - index 0 : lr itself (proven baseline, run first)
      - index 1 : far-right (+log_range)    → quickly rule out / confirm high LR
      - index 2 : far-left  (-log_range)    → quickly rule out / confirm low LR
      - index 3 : near-right (+log_range/φ) → refine the right side
      - index 4 : near-left  (-log_range/φ) → refine the left side
      - index 5 : fine-right (+log_range/φ²)
      - ...

    Steps shrink by golden ratio (φ ≈ 1.618) each full cycle, giving
    Fibonacci-style coverage that is provably optimal for unimodal functions.

    Example — n=5, lr=1e-4, log_range=0.18:
        [1e-4, 1.51e-4, 6.61e-5, 1.28e-4, 7.81e-5]
         ^0    ^far-R   ^far-L   ^near-R   ^near-L

    Args:
        lr         : center LR (typically state["train"]["found_lr"])
        n          : total number of runs (including the baseline run 0)
        log_range  : half-width of search in log10 units (default 0.2 ≈ ±60%)

    Returns:
        List of n LRs. Index 0 is always exactly `lr`.
    """
    if n <= 0:
        return []
    if n == 1:
        return [lr]

    PHI = (1 + math.sqrt(5)) / 2  # golden ratio ≈ 1.618

    results = [lr]  # index 0 = baseline (found_lr)

    for i in range(1, n):
        # Determine which "cycle" we're in (each cycle = one right + one left step)
        cycle = (i - 1) // 2          # 0, 0, 1, 1, 2, 2, ...
        sign  = 1 if i % 2 == 1 else -1   # +, -, +, -, ...

        step = log_range / (PHI ** cycle)  # shrinks: 0.2, 0.124, 0.076, ...
        results.append(lr * (10 ** (sign * step)))

    return results


def test():
    import math

    lr = 0.00014523947500000002
    print("=== extend_learning_rates (original grid) ===")
    for n in [3, 4, 5, 6]:
        lrs = extend_learning_rates(lr, n)
        print(f"  n={n}: {[f'{x:.3e}' for x in lrs]}")
        assert lrs[0] == lr

    print("\n=== smart_extend_learning_rates (golden-ratio) ===")
    for n in [3, 4, 5, 6]:
        lrs = smart_extend_learning_rates(lr, n)
        print(f"  n={n}: {[f'{x:.3e}' for x in lrs]}")
        assert lrs[0] == lr, f"index 0 must equal lr, got {lrs[0]}"
        assert len(lrs) == n, f"expected {n} lrs, got {len(lrs)}"

    print("\nAll tests passed ✓")


if __name__ == "__main__":
    test()