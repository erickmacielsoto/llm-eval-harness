from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

from .runner import Record, RunMeta
from .stats import mcnemar_exact, paired_bootstrap_diff, percentile, wilson_interval


@dataclass(frozen=True)
class Summary:
    model: str
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    parse_failures: int
    hit_token_limit: int | None
    latency_p50_s: float
    latency_p95_s: float
    completion_tokens_mean: float
    throughput_tok_s: float
    wall_time_s: float
    gpu_cost_usd: float
    cost_per_1k_correct_usd: float | None


@dataclass(frozen=True)
class Comparison:
    model_a: str
    model_b: str
    accuracy_diff: float
    diff_ci_low: float
    diff_ci_high: float
    only_a_correct: int
    only_b_correct: int
    mcnemar_p: float
    significant: bool


def summarize(meta: RunMeta, records: Sequence[Record], usd_per_gpu_hour: float) -> Summary:
    n = len(records)
    correct = sum(r.correct for r in records)
    ci = wilson_interval(correct, n)
    latencies = [r.latency_s for r in records]
    tokens = sum(r.completion_tokens for r in records)
    cost = meta.wall_time_s / 3600 * usd_per_gpu_hour if meta.uses_gpu else 0.0
    return Summary(
        model=meta.model,
        n=n,
        accuracy=correct / n if n else 0.0,
        ci_low=ci.low,
        ci_high=ci.high,
        parse_failures=sum(r.prediction is None for r in records),
        hit_token_limit=(
            sum(r.completion_tokens >= meta.max_new_tokens for r in records) if meta.max_new_tokens else None
        ),
        latency_p50_s=percentile(latencies, 0.5),
        latency_p95_s=percentile(latencies, 0.95),
        completion_tokens_mean=tokens / n if n else 0.0,
        throughput_tok_s=tokens / meta.wall_time_s if meta.wall_time_s > 0 else 0.0,
        wall_time_s=meta.wall_time_s,
        gpu_cost_usd=cost,
        cost_per_1k_correct_usd=cost / correct * 1000 if correct else None,
    )


def compare(a: Sequence[Record], b: Sequence[Record], alpha: float = 0.05) -> Comparison:
    by_id_b = {r.example_id: r for r in b}
    shared = [r for r in a if r.example_id in by_id_b]
    if not shared:
        raise ValueError("runs share no examples")
    xs = [r.correct for r in shared]
    ys = [by_id_b[r.example_id].correct for r in shared]
    diff, ci = paired_bootstrap_diff(xs, ys, alpha=alpha)
    only_a, only_b, p = mcnemar_exact(xs, ys)
    return Comparison(
        model_a=shared[0].model,
        model_b=by_id_b[shared[0].example_id].model,
        accuracy_diff=diff,
        diff_ci_low=ci.low,
        diff_ci_high=ci.high,
        only_a_correct=only_a,
        only_b_correct=only_b,
        mcnemar_p=p,
        significant=p < alpha,
    )


def build_report(
    runs: Sequence[tuple[RunMeta, Sequence[Record]]], usd_per_gpu_hour: float
) -> tuple[list[Summary], list[Comparison]]:
    summaries = [summarize(meta, records, usd_per_gpu_hour) for meta, records in runs]
    comparisons = [compare(ra, rb) for (_, ra), (_, rb) in combinations(runs, 2)]
    return summaries, comparisons


def to_json(summaries: Sequence[Summary], comparisons: Sequence[Comparison]) -> dict[str, Any]:
    return {"summaries": [asdict(s) for s in summaries], "comparisons": [asdict(c) for c in comparisons]}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def to_markdown(summaries: Sequence[Summary], comparisons: Sequence[Comparison], usd_per_gpu_hour: float) -> str:
    lines = [
        (
            "| Model | n | Accuracy | 95% CI (Wilson) | Parse fail | Hit token limit | p50 s/item | p95 s/item "
            "| Mean out tokens | Tok/s | Wall time s | GPU cost USD | USD / 1k correct |"
        ),
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        per_k = "n/a" if s.cost_per_1k_correct_usd is None else f"{s.cost_per_1k_correct_usd:.4f}"
        limit = "n/a" if s.hit_token_limit is None else str(s.hit_token_limit)
        lines.append(
            f"| `{s.model}` | {s.n} | {_pct(s.accuracy)} | {_pct(s.ci_low)} to {_pct(s.ci_high)} "
            f"| {s.parse_failures} | {limit} | {s.latency_p50_s:.3f} | {s.latency_p95_s:.3f} "
            f"| {s.completion_tokens_mean:.0f} | {s.throughput_tok_s:.0f} | {s.wall_time_s:.1f} "
            f"| {s.gpu_cost_usd:.4f} | {per_k} |"
        )
    lines += [
        "",
        f"GPU cost assumes USD {usd_per_gpu_hour:.2f} per GPU-hour applied to wall time.",
        "",
        (
            "| A | B | Acc(A) - Acc(B) | 95% CI (paired bootstrap) | Only A correct | Only B correct "
            "| McNemar exact p | Significant (alpha=0.05) |"
        ),
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| `{c.model_a}` | `{c.model_b}` | {100 * c.accuracy_diff:+.1f} pp "
            f"| {100 * c.diff_ci_low:+.1f} to {100 * c.diff_ci_high:+.1f} pp | {c.only_a_correct} "
            f"| {c.only_b_correct} | {c.mcnemar_p:.2g} | {'yes' if c.significant else 'no'} |"
        )
    return "\n".join(lines) + "\n"
