from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends import make_backend
from .report import build_report, to_json, to_markdown
from .runner import load_run, run, save_run
from .tasks import load_gsm8k


def _run(args: argparse.Namespace) -> None:
    examples = load_gsm8k(args.n, seed=args.seed, cache=Path(args.cache))
    backend = make_backend(args.model, max_new_tokens=args.max_new_tokens)
    meta, records = run(
        backend,
        examples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        on_progress=lambda done, total: print(f"{backend.name}: {done}/{total}", file=sys.stderr, flush=True),
    )
    save_run(Path(args.out), meta, records)
    accuracy = sum(r.correct for r in records) / len(records)
    print(f"{meta.model}: accuracy={accuracy:.3f} n={meta.n} wall={meta.wall_time_s:.1f}s -> {args.out}")


def _report(args: argparse.Namespace) -> None:
    runs = [load_run(Path(p)) for p in args.runs]
    summaries, comparisons = build_report(runs, args.usd_per_gpu_hour)
    markdown = to_markdown(summaries, comparisons, args.usd_per_gpu_hour)
    if args.markdown:
        Path(args.markdown).write_text(markdown, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(to_json(summaries, comparisons), indent=2), encoding="utf-8")
    print(markdown)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evalh", description="Evaluate LLMs on a GSM8K subset.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one model and store per-example records as JSONL.")
    run_p.add_argument("--model", required=True, help="hf:<id>, ollama:<model> or baseline:last-number")
    run_p.add_argument("--n", type=int, default=200)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--batch-size", type=int, default=16)
    run_p.add_argument("--max-new-tokens", type=int, default=384)
    run_p.add_argument("--cache", default="data/gsm8k_test_subset.jsonl")
    run_p.add_argument("--out", required=True)
    run_p.set_defaults(func=_run)

    rep_p = sub.add_parser("report", help="Summarize runs and compare every pair.")
    rep_p.add_argument("runs", nargs="+")
    rep_p.add_argument("--usd-per-gpu-hour", type=float, default=0.20)
    rep_p.add_argument("--markdown")
    rep_p.add_argument("--json")
    rep_p.set_defaults(func=_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
