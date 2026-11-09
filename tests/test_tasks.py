import pytest

from evalh.tasks import extract_answer, gold_answer, is_correct, normalize_number, sample_examples


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1,234", "1234"), ("$18.00", "18"), ("0.50", "0.5"), ("-3.", "-3"), ("abc", None)],
)
def test_normalize_number(raw: str, expected: str | None) -> None:
    assert normalize_number(raw) == expected


def test_gold_answer_reads_marker() -> None:
    assert gold_answer("She has 3 + 4 = <<3+4=7>>7 apples.\n#### 7") == "7"


def test_gold_answer_requires_marker() -> None:
    with pytest.raises(ValueError):
        gold_answer("no marker here")


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("3 apples plus 4 is 7.\nThe answer is: 7", "7"),
        ("Total cost is $1,250.\nThe answer is: **$1,250**", "1250"),
        ("So the result is \\boxed{42}.", "42"),
        ("First 10, then 12, so 22 in total", "22"),
        ("I don't know", None),
    ],
)
def test_extract_answer(completion: str, expected: str | None) -> None:
    assert extract_answer(completion) == expected


def test_final_answer_pattern_wins_over_trailing_numbers() -> None:
    assert extract_answer("The answer is: 18\nCheck: 9 * 2 = 18, 3 steps") == "18"


def test_is_correct_numeric_equivalence() -> None:
    assert is_correct("18", "18")
    assert is_correct("0.5", "0.50")
    assert not is_correct(None, "1")
    assert not is_correct("17", "18")


def test_sample_examples_is_deterministic() -> None:
    rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(50)]
    first = sample_examples(rows, 5, seed=1)
    assert first == sample_examples(rows, 5, seed=1)
    assert [e.id for e in first] != [e.id for e in sample_examples(rows, 5, seed=2)]
