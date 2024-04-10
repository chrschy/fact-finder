import textdistance as td

from fact_finder.evaluator.score.score import Score


class LevenshteinScore(Score):
    """
    The Levenshtein distance between two words is the minimum number of single-character edits (insertions, deletions or substitutions) required to change one word into the other.
    """

    def compare(self, text_a: str, text_b: str) -> float:
        return float(td.levenshtein.normalized_similarity(text_a, text_b))
