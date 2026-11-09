from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends import Backend
from .tasks import Example, build_messages, extract_answer, is_correct


@dataclass(frozen=True)
class Record:
    model: str
    example_id: str
    gold: str
    prediction: str | None
    correct: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    output: str


@dataclass(frozen=True)
class RunMeta:
    model: str
    n: int
    batch_size: int
    wall_time_s: float
    uses_gpu: bool
    max_new_tokens: int | None = None


def run(
    backend: Backend,
    examples: Sequence[Example],
    batch_size: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
    max_new_tokens: int | None = None,
) -> tuple[RunMeta, list[Record]]:
    records: list[Record] = []
    start = time.perf_counter()
    for offset in range(0, len(examples), batch_size):
        chunk = examples[offset : offset + batch_size]
        generations = backend.generate([build_messages(e.question) for e in chunk])
        for example, gen in zip(chunk, generations, strict=True):
            prediction = extract_answer(gen.text)
            records.append(
                Record(
                    model=backend.name,
                    example_id=example.id,
                    gold=example.answer,
                    prediction=prediction,
                    correct=is_correct(prediction, example.answer),
                    latency_s=gen.latency_s,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    output=gen.text,
                )
            )
        if on_progress:
            on_progress(len(records), len(examples))
    meta = RunMeta(
        model=backend.name,
        n=len(records),
        batch_size=batch_size,
        wall_time_s=time.perf_counter() - start,
        uses_gpu=backend.uses_gpu,
        max_new_tokens=max_new_tokens if backend.uses_gpu else None,
    )
    return meta, records


def save_run(path: Path, meta: RunMeta, records: Sequence[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"meta": asdict(meta)})] + [json.dumps(asdict(r)) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_run(path: Path) -> tuple[RunMeta, list[Record]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = RunMeta(**json.loads(lines[0])["meta"])
    return meta, [Record(**json.loads(line)) for line in lines[1:] if line]
