import json
from typing import Dict, Any

import pandas as pd
from tqdm import tqdm

import fact_finder.config.primekg_config as graph_config
from evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluator.answer_evaluator import AnswerEvaluator
from fact_finder.evaluator.evaluator.cypher_query_generation_evaluator import CypherQueryGenerationEvaluator
from fact_finder.evaluator.score.bleu_score import BleuScore
from fact_finder.evaluator.score.difflib_score import DifflibScore
from fact_finder.evaluator.score.embedding_score import EmbeddingScore
from fact_finder.evaluator.score.levenshtein_score import LevenshteinScore
from fact_finder.utils import load_chat_model


class Evaluation:

    def __init__(self):
        self.chat_model = load_chat_model()
        self.chain = graph_config.build_chain(
            self.chat_model, ["--normalized_graph", "--use_entity_detection_preprocessing"]
        )
        self.eval_samples = self.eval_samples("evaluation_samples.json")
        self.evaluators = [CypherQueryGenerationEvaluator(), AnswerEvaluator()]
        self.scores = [BleuScore(), DifflibScore(), EmbeddingScore(), LevenshteinScore()]

    def run(self):
        chain_results = self.run_chain()
        evaluation = self.evaluate(chain_results)
        return evaluation

    def evaluate(self, chain_results):
        evaluation = {}
        print("Evaluating...")
        for evaluator in tqdm(self.evaluators):
            evaluation[evaluator.__class__.__name__] = evaluator.evaluate(self.eval_samples, chain_results, self.scores)
        return evaluation

    def run_chain(self):
        results = []
        print("Running Chain...")
        for eval_sample in tqdm(self.eval_samples):
            inputs = {"question": eval_sample.question}
            result = self.chain.invoke(inputs)
            results.append(result)
        return results

    def eval_samples(self, file_path: str):
        with open(file_path) as file:
            data = json.load(file)
        eval_samples = [EvaluationSample(**d) for d in data]
        return eval_samples

    def save_as_excel(self, result: Dict[str, Any], path: str = "hello.xlsx"):
        df = pd.DataFrame(result)
        df.to_excel(path)


if __name__ == "__main__":
    evaluation = Evaluation()
    results = evaluation.run()
    print(results)
