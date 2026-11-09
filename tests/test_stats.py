import math

import pytest

from evalh.stats import mcnemar_exact, paired_bootstrap_diff, percentile, wilson_interval


def test_wilson_matches_reference_values() -> None:
    ci = wilson_interval(50, 100)
    assert ci.low == pytest.approx(0.4038, abs=1e-4)
    assert ci.high == pytest.approx(0.5962, abs=1e-4)


def test_wilson_handles_extremes() -> None:
    assert wilson_interval(0, 20).low == 0.0
    assert wilson_interval(20, 20).high == pytest.approx(1.0)
    assert wilson_interval(0, 0).high == 0.0


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert math.isnan(percentile([], 0.5))


def test_mcnemar_exact_known_value() -> None:
    a = [True] * 10 + [False] * 2 + [True] * 30
    b = [False] * 10 + [True] * 2 + [True] * 30
    only_a, only_b, p = mcnemar_exact(a, b)
    assert (only_a, only_b) == (10, 2)
    assert p == pytest.approx(0.0386, abs=1e-4)


def test_mcnemar_identical_runs() -> None:
    assert mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)


def test_mcnemar_requires_pairs() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


def test_paired_bootstrap_contains_observed_difference() -> None:
    a = [True] * 70 + [False] * 30
    b = [True] * 50 + [False] * 50
    diff, ci = paired_bootstrap_diff(a, b, iterations=2000)
    assert diff == pytest.approx(0.2)
    assert ci.low < 0.2 < ci.high
    assert ci.low > 0
