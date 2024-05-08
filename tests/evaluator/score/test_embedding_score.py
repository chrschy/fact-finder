import pytest
from fact_finder.evaluator.score.embedding_score import EmbeddingScore

# todo mock the model behaviour


@pytest.fixture
def scorer():
    return EmbeddingScore()


def test_exact_match(scorer):
    text_a = "This is a test."
    text_b = "This is a test."
    score = scorer.compare(text_a, text_b)
    assert score == pytest.approx(1.0)


def test_no_match(scorer):
    text_a = "This is a test."
    text_b = "Hello"
    score = scorer.compare(text_a, text_b)
    assert score < 0.2


def test_partial_match(scorer):
    text_a = "This is a test."
    text_b = "This is a different test."
    score = scorer.compare(text_a, text_b)
    assert score > 0.0
    assert score < 1.0


def test_similarity(scorer):
    text_a = "I had a great vacation."
    text_b = "My holiday was fantastic."
    score = scorer.compare(text_a, text_b)
    assert score > 0.5
