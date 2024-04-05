import json

from tqdm import tqdm

import fact_finder.config.primekg_config as graph_config
from evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluator.answer_evaluator import AnswerEvaluator
from fact_finder.evaluator.evaluator.cypher_query_generation_evaluator import CypherQueryGenerationEvaluator
from fact_finder.evaluator.score.bleu_score import BleuScore
from fact_finder.evaluator.score.difflib_score import DifflibScore
from fact_finder.evaluator.score.embedding_score import EmbeddingScore
from fact_finder.utils import load_chat_model


def _eval_samples(file_path: str = "evaluation_samples.json"):
    with open(file_path) as file:
        data = json.load(file)
    eval_samples = [EvaluationSample(**d) for d in data]
    return eval_samples


if __name__ == "__main__":
    chat_model = load_chat_model()
    chain = graph_config.build_chain(chat_model, ["--normalized_graph", "--use_entity_detection_preprocessing"])
    eval_samples = _eval_samples()
    results = []
    print("Running Chain...")
    for eval_sample in tqdm(eval_samples):
        inputs = {"question": eval_sample.question}
        result = chain.invoke(inputs)
        results.append(result)
    evaluators = [CypherQueryGenerationEvaluator(), AnswerEvaluator()]
    scores = [BleuScore(), DifflibScore(), EmbeddingScore()]
    evaluation = {}
    print("Evaluating...")
    for evaluator in tqdm(evaluators):
        evaluation[evaluator.__class__.__name__] = evaluator.evaluate(eval_samples, results, scores)
    breakpoint()
