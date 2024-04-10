import difflib

from fact_finder.evaluator.score.score import Score


class DifflibScore(Score):
    """
    Finds the longest matching subsequence.
    Then, to the left and right of the longest matching subsequence, it again finds the longest matching subsequence.
    Omits "junk" elements such as white spaces.
    """

    def compare(self, text_a: str, text_b: str) -> float:
        # set autojunk=False if you don't want junk to be detected automatically.
        sequence_matcher = difflib.SequenceMatcher(None, text_a, text_b, autojunk=True)
        similarity_score = sequence_matcher.ratio()
        return similarity_score
