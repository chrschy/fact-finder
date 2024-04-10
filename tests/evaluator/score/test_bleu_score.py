from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

from fact_finder.evaluator.score.bleu_score import BleuScore


def test_compare_different_sentences():
    score = BleuScore()
    text_a = "This is a test sentence."
    text_b = "This is another test sentence."
    expected_score = sentence_bleu([word_tokenize(text_a)], word_tokenize(text_b))
    assert score.compare(text_a, text_b) == expected_score


def test_compare_same_sentences():
    score = BleuScore()
    text_a = "This is a test sentence."
    text_b = "This is a test sentence."
    expected_score = sentence_bleu([word_tokenize(text_a)], word_tokenize(text_b))
    score = score.compare(text_a, text_b)
    assert score == expected_score
    assert score == 1.0
