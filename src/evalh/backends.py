from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

Messages = list[dict[str, str]]


@dataclass(frozen=True)
class Generation:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


class Backend(Protocol):
    name: str
    uses_gpu: bool

    def generate(self, batch: list[Messages]) -> list[Generation]: ...


class LastNumberBaseline:
    """Answers with the last number that appears in the problem statement."""

    name = "baseline:last-number"
    uses_gpu = False

    def generate(self, batch: list[Messages]) -> list[Generation]:
        out: list[Generation] = []
        for messages in batch:
            start = time.perf_counter()
            numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", messages[-1]["content"])
            text = f"The answer is: {numbers[-1] if numbers else 0}"
            out.append(Generation(text, time.perf_counter() - start, 0, 0))
        return out


class HFBackend:
    uses_gpu = True

    def __init__(self, model_id: str, max_new_tokens: int = 384, dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model_id}"
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, batch: list[Messages]) -> list[Generation]:
        torch = self._torch
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch
        ]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per_item = (time.perf_counter() - start) / len(batch)
        new_tokens = output[:, enc["input_ids"].shape[1] :]
        results: list[Generation] = []
        for i, row in enumerate(new_tokens):
            completion_ids = [t for t in row.tolist() if t != self.tokenizer.pad_token_id]
            results.append(
                Generation(
                    text=self.tokenizer.decode(completion_ids, skip_special_tokens=True),
                    latency_s=per_item,
                    prompt_tokens=int(enc["attention_mask"][i].sum()),
                    completion_tokens=len(completion_ids),
                )
            )
        return results


class OllamaBackend:
    uses_gpu = True

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", max_new_tokens: int = 384) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.host = host.rstrip("/")
        self.max_new_tokens = max_new_tokens

    def _chat(self, messages: Messages) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_predict": self.max_new_tokens},
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            payload: dict[str, Any] = json.load(response)
        return payload

    def generate(self, batch: list[Messages]) -> list[Generation]:
        out: list[Generation] = []
        for messages in batch:
            start = time.perf_counter()
            payload = self._chat(messages)
            out.append(
                Generation(
                    text=payload["message"]["content"],
                    latency_s=time.perf_counter() - start,
                    prompt_tokens=int(payload.get("prompt_eval_count", 0)),
                    completion_tokens=int(payload.get("eval_count", 0)),
                )
            )
        return out


def make_backend(spec: str, max_new_tokens: int = 384) -> Backend:
    kind, _, target = spec.partition(":")
    if kind == "baseline" and target == "last-number":
        return LastNumberBaseline()
    if kind == "hf" and target:
        return HFBackend(target, max_new_tokens=max_new_tokens)
    if kind == "ollama" and target:
        return OllamaBackend(target, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unknown backend spec {spec!r}; use hf:<id>, ollama:<model> or baseline:last-number")
