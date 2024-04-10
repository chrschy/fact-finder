from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

from fact_finder.evaluator.score.score import Score


class BleuScore(Score):
    """
    1. punish length differences
    2. compute n-gram-overlap for n=1,...,4 --> how many bi-grams / tri-grams / ... match in the sequences
    e.g. a bi-gram in a text sequence is two adjacent tokens
    If any count for the different n-grams is 0, the BLEU score is also 0. The smoothing function avoids that.
    """

    def compare(self, text_a: str, text_b: str) -> float:
        tokens_a = word_tokenize(text_a)
        tokens_b = word_tokenize(text_b)
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu([tokens_a], tokens_b, smoothing_function=smoothing_function)
        return bleu_score
