import pytest

from fact_finder.evaluator.score.difflib_score import DifflibScore


@pytest.fixture
def scorer():
    return DifflibScore()


def test_exact_match(scorer):
    text_a = "example"
    text_b = "example"
    score = scorer.compare(text_a, text_b)
    assert score == 1.0


def test_no_match(scorer):
    text_a = "example"
    text_b = "different"
    score = scorer.compare(text_a, text_b)
    assert score == 0.25


def test_partial_match(scorer):
    text_a = "example"
    text_b = "sample"
    score = scorer.compare(text_a, text_b)
    assert score == 0.7692307692307693


def test_case_sensitivity(scorer):
    text_a = "Example"
    text_b = "example"
    score = scorer.compare(text_a, text_b)
    assert score == 0.8571428571428571
