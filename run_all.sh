#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python}
N=${N:-300}
MODELS=(baseline:last-number hf:Qwen/Qwen2.5-0.5B-Instruct hf:Qwen/Qwen2.5-1.5B-Instruct)
if [[ "${WITH_OLLAMA:-0}" == "1" ]]; then
  MODELS+=(ollama:qwen2.5:1.5b)
fi
mkdir -p results
for spec in "${MODELS[@]}"; do
  name=$(echo "$spec" | tr ':/' '__')
  if [[ ! -f "results/$name.jsonl" ]]; then
    "$PY" -m evalh.cli run --model "$spec" --n "$N" --out "results/$name.jsonl"
  fi
done
"$PY" -m evalh.cli report results/*.jsonl --markdown results/report.md --json results/report.json
