from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

SYSTEM_PROMPT = "You are a careful math tutor."
INSTRUCTION = (
    "Solve the following grade-school math problem. Think step by step, "
    "then finish with a final line of the form 'The answer is: <number>'."
)

_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"
_FINAL_PATTERNS = (
    re.compile(rf"answer is[:\s]*\**\s*({_NUMBER})", re.IGNORECASE),
    re.compile(rf"\\boxed\{{\s*({_NUMBER})"),
    re.compile(rf"####\s*({_NUMBER})"),
)
_ANY_NUMBER = re.compile(_NUMBER)


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str


def normalize_number(raw: str) -> str | None:
    cleaned = raw.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def gold_answer(solution: str) -> str:
    match = re.search(r"####\s*(.+)$", solution.strip())
    if not match:
        raise ValueError("GSM8K solution without '####' marker")
    normalized = normalize_number(match.group(1))
    if normalized is None:
        raise ValueError(f"Non-numeric gold answer: {match.group(1)!r}")
    return normalized


def extract_answer(completion: str) -> str | None:
    for pattern in _FINAL_PATTERNS:
        matches = pattern.findall(completion)
        if matches:
            return normalize_number(matches[-1])
    numbers = _ANY_NUMBER.findall(completion)
    return normalize_number(numbers[-1]) if numbers else None


def is_correct(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    try:
        return abs(float(prediction) - float(gold)) < 1e-6
    except ValueError:
        return prediction == gold


def build_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INSTRUCTION}\n\nProblem: {question}"},
    ]


def sample_examples(rows: list[dict[str, str]], n: int, seed: int) -> list[Example]:
    indexed = list(enumerate(rows))
    random.Random(seed).shuffle(indexed)
    return [
        Example(id=f"gsm8k-test-{i}", question=row["question"], answer=gold_answer(row["answer"]))
        for i, row in indexed[:n]
    ]


def load_gsm8k(n: int, seed: int = 0, split: str = "test", cache: Path | None = None) -> list[Example]:
    if cache is not None and cache.exists():
        cached = [Example(**json.loads(line)) for line in cache.read_text(encoding="utf-8").splitlines()]
        if len(cached) >= n:
            return cached[:n]
    from datasets import load_dataset

    rows = [dict(row) for row in load_dataset("openai/gsm8k", "main", split=split)]
    examples = sample_examples(rows, n, seed)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(json.dumps(asdict(e)) for e in examples) + "\n", encoding="utf-8")
    return examples
