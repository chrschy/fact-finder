from sentence_transformers import SentenceTransformer, util

from fact_finder.evaluator.score.score import Score


class EmbeddingScore(Score):
    """
    Use Sentence Transformers to
    1. embed the text
    2. compare the vectors (cosine similarity)
    """

    def __init__(self):
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def compare(self, text_a: str, text_b: str) -> float:
        embedding_1 = self._model.encode(text_a, convert_to_tensor=True)
        embedding_2 = self._model.encode(text_b, convert_to_tensor=True)
        similarity_tensor = util.pytorch_cos_sim(embedding_1, embedding_2)[0]
        return float(similarity_tensor)
