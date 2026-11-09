from pathlib import Path

from evalh.backends import Generation, LastNumberBaseline, Messages, make_backend
from evalh.report import build_report, to_markdown
from evalh.runner import load_run, run, save_run
from evalh.tasks import Example

EXAMPLES = [
    Example(id="a", question="Tom has 3 apples and buys 4 more. How many?", answer="7"),
    Example(id="b", question="A box holds 6 eggs. How many eggs are in 2 boxes?", answer="12"),
    Example(id="c", question="Sara reads 5 pages a day for 3 days.", answer="15"),
]


class OracleBackend:
    name = "fake:oracle"
    uses_gpu = True

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def generate(self, batch: list[Messages]) -> list[Generation]:
        out = []
        for messages in batch:
            answer = next(v for k, v in self.answers.items() if k in messages[-1]["content"])
            out.append(Generation(f"Reasoning...\nThe answer is: {answer}", 0.5, 10, 20))
        return out


def test_run_scores_and_roundtrips(tmp_path: Path) -> None:
    backend = OracleBackend({"apples": "7", "eggs": "12", "pages": "14"})
    meta, records = run(backend, EXAMPLES, batch_size=2)
    assert [r.correct for r in records] == [True, True, False]
    path = tmp_path / "run.jsonl"
    save_run(path, meta, records)
    loaded_meta, loaded = load_run(path)
    assert loaded_meta == meta
    assert loaded == records


def test_baseline_uses_last_number_in_question() -> None:
    _, records = run(LastNumberBaseline(), EXAMPLES)
    assert [r.prediction for r in records] == ["4", "2", "3"]
    assert make_backend("baseline:last-number").name == "baseline:last-number"


def test_report_contains_summary_and_comparison() -> None:
    meta_a, rec_a = run(OracleBackend({"apples": "7", "eggs": "12", "pages": "15"}), EXAMPLES)
    meta_b, rec_b = run(LastNumberBaseline(), EXAMPLES)
    summaries, comparisons = build_report([(meta_a, rec_a), (meta_b, rec_b)], usd_per_gpu_hour=1.0)
    assert summaries[0].accuracy == 1.0
    assert summaries[1].gpu_cost_usd == 0.0
    assert comparisons[0].only_a_correct == 3
    markdown = to_markdown(summaries, comparisons, 1.0)
    assert "`fake:oracle`" in markdown
    assert "McNemar" in markdown


def test_summary_counts_completions_that_hit_the_token_limit() -> None:
    backend = OracleBackend({"apples": "7", "eggs": "12", "pages": "15"})
    meta, records = run(backend, EXAMPLES, max_new_tokens=20)
    summaries, _ = build_report([(meta, records)], usd_per_gpu_hour=0.0)
    assert summaries[0].hit_token_limit == 3
    meta_b, records_b = run(LastNumberBaseline(), EXAMPLES, max_new_tokens=20)
    assert build_report([(meta_b, records_b)], 0.0)[0][0].hit_token_limit is None
