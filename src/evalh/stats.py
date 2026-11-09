from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> Interval:
    if n == 0:
        return Interval(0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Two-sided exact McNemar test on paired outcomes. Returns (only_a, only_b, p_value)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return only_a, only_b, min(1.0, 2 * tail)


def paired_bootstrap_diff(
    a: Sequence[bool], b: Sequence[bool], iterations: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, Interval]:
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and of equal length")
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    samples = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    observed = sum(diffs) / n
    return observed, Interval(percentile(samples, alpha / 2), percentile(samples, 1 - alpha / 2))
