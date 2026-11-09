| Model | n | Accuracy | 95% CI (Wilson) | Parse fail | Hit token limit | p50 s/item | p95 s/item | Mean out tokens | Tok/s | Wall time s | GPU cost USD | USD / 1k correct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline:last-number` | 300 | 2.7% | 1.4% to 5.2% | 0 | n/a | 0.000 | 0.000 | 0 | 0 | 0.0 | 0.0000 | 0.0000 |
| `hf:Qwen/Qwen2.5-0.5B-Instruct` | 300 | 37.0% | 31.7% to 42.6% | 0 | 58 | 0.954 | 0.999 | 295 | 311 | 284.3 | 0.0158 | 0.1423 |
| `hf:Qwen/Qwen2.5-1.5B-Instruct` | 300 | 66.0% | 60.5% to 71.1% | 0 | 42 | 1.106 | 1.761 | 281 | 237 | 355.3 | 0.0197 | 0.0997 |
| `ollama:qwen2.5:1.5b` | 300 | 63.3% | 57.7% to 68.6% | 0 | 50 | 1.231 | 1.602 | 282 | 206 | 409.6 | 0.0228 | 0.1198 |

GPU cost assumes USD 0.20 per GPU-hour applied to wall time.

| A | B | Acc(A) - Acc(B) | 95% CI (paired bootstrap) | Only A correct | Only B correct | McNemar exact p | Significant (alpha=0.05) |
|---|---|---:|---|---:|---:|---:|---|
| `baseline:last-number` | `hf:Qwen/Qwen2.5-0.5B-Instruct` | -34.3 pp | -40.0 to -28.7 pp | 3 | 106 | 6.7e-28 | yes |
| `baseline:last-number` | `hf:Qwen/Qwen2.5-1.5B-Instruct` | -63.3 pp | -68.7 to -57.7 pp | 0 | 190 | 1.3e-57 | yes |
| `baseline:last-number` | `ollama:qwen2.5:1.5b` | -60.7 pp | -66.3 to -54.7 pp | 3 | 185 | 5.6e-51 | yes |
| `hf:Qwen/Qwen2.5-0.5B-Instruct` | `hf:Qwen/Qwen2.5-1.5B-Instruct` | -29.0 pp | -35.3 to -22.7 pp | 15 | 102 | 4.4e-17 | yes |
| `hf:Qwen/Qwen2.5-0.5B-Instruct` | `ollama:qwen2.5:1.5b` | -26.3 pp | -32.7 to -19.7 pp | 20 | 99 | 9.2e-14 | yes |
| `hf:Qwen/Qwen2.5-1.5B-Instruct` | `ollama:qwen2.5:1.5b` | +2.7 pp | -3.0 to +8.0 pp | 39 | 31 | 0.4 | no |
