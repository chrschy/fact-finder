import json
from typing import Dict, Any, Optional
from typing import List

import pandas as pd
from langchain.chains.base import Chain
from langchain_core.language_models import BaseChatModel
from tqdm import tqdm

import fact_finder.config.primekg_config as graph_config
from evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluator.answer_evaluator import AnswerEvaluator
from fact_finder.evaluator.evaluator.cypher_query_generation_evaluator import CypherQueryGenerationEvaluator
from fact_finder.evaluator.evaluator.evaluator import Evaluator
from fact_finder.evaluator.score.bleu_score import BleuScore
from fact_finder.evaluator.score.difflib_score import DifflibScore
from fact_finder.evaluator.score.embedding_score import EmbeddingScore
from fact_finder.evaluator.score.levenshtein_score import LevenshteinScore
from fact_finder.evaluator.score.score import Score
from fact_finder.utils import load_chat_model


class Evaluation:

    def __init__(
        self,
        chat_model: BaseChatModel = None,
        chain: Chain = None,
        chain_args: List[str] = ["--normalized_graph", "--use_entity_detection_preprocessing"],
        eval_path: str = "evaluation_samples.json",
        evaluators: List[Evaluator] = [CypherQueryGenerationEvaluator(), AnswerEvaluator()],
        scores: List[Score] = [BleuScore(), DifflibScore(), EmbeddingScore(), LevenshteinScore()],
    ):
        if not chat_model:
            self.chat_model = load_chat_model()
        if not chain:
            self.chain = graph_config.build_chain(
                model=self.chat_model, combine_output_with_sematic_scholar=True, args=chain_args
            )
        self.eval_samples = self.eval_samples(eval_path)
        self.evaluators = evaluators
        self.scores = scores

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
