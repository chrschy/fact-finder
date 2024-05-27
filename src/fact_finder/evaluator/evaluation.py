import os.path
from typing import Dict, Any
from typing import List, Union

import pandas as pd
from langchain.chains.base import Chain
from langchain_core.language_models import BaseChatModel
from tqdm import tqdm

import fact_finder.config.primekg_config as graph_config
from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluation_samples import manual_samples
from fact_finder.evaluator.score.bleu_score import BleuScore
from fact_finder.evaluator.score.difflib_score import DifflibScore
from fact_finder.evaluator.score.embedding_score import EmbeddingScore
from fact_finder.evaluator.score.levenshtein_score import LevenshteinScore
from fact_finder.evaluator.score.score import Score
from fact_finder.evaluator.set_evaluator.set_evaluator import SetEvaluator
from fact_finder.evaluator.string_evaluator.string_evaluator import StringEvaluator
from fact_finder.evaluator.util import load_pickle, save_pickle
from fact_finder.utils import load_chat_model


class Evaluation:

    def __init__(
        self,
        evaluators: List[Union[StringEvaluator, SetEvaluator]],
        chat_model: BaseChatModel = None,
        chain: Chain = None,
        chain_args: List[str] = ["--normalized_graph"],  # , "--use_entity_detection_preprocessing"],
        scores: List[Score] = [BleuScore(), DifflibScore(), EmbeddingScore(), LevenshteinScore()],
        limit_of_samples: int = None,
        idx_list_of_samples: List[int] = None,
    ):
        self.idx_list_of_samples = idx_list_of_samples
        if not chat_model:
            self.chat_model = load_chat_model()
        if not chain:
            self.chain = graph_config.build_chain(
                model=self.chat_model, combine_output_with_sematic_scholar=False, args=chain_args
            )
        self.eval_samples = self.eval_samples(limit_of_samples=limit_of_samples)
        self.evaluators = evaluators
        self.scores = scores

    def run(self, save_as_excel: bool = False, cache_path="cached_results/chain_results.pickle"):
        chain_results = self.run_chain(cache_path)
        results = self.evaluate(chain_results)
        if save_as_excel:
            self.save_as_excel(results)
        return results

    def evaluate(self, chain_results) -> Dict[str, Any]:
        evaluation = {}
        print("Evaluating...")
        for evaluator in self.evaluators:
            evaluation[evaluator.__class__.__name__] = evaluator.evaluate(
                evaluation_samples=self.eval_samples, chain_results=chain_results, scores=self.scores
            )
        return evaluation

    def run_chain(self, cache_path: str):
        results = []
        print("Running Chain...")
        if os.path.exists(cache_path):
            return load_pickle(cache_path)
        for eval_sample in tqdm(self.eval_samples):
            inputs = {"question": eval_sample.question}
            try:
                result = self.chain.invoke(inputs)
            except Exception as e:
                print(e)
                result = {}
            results.append(result)
        save_pickle(results, cache_path)
        return results

    def eval_samples(self, limit_of_samples: int = None):
        eval_samples = []
        samples = manual_samples
        if self.idx_list_of_samples:
            samples = [samples[i] for i in self.idx_list_of_samples]
        for sample in samples[:limit_of_samples]:
            eval_sample = EvaluationSample(
                question=sample["question"],
                cypher_query=sample["expected_cypher"],
                expected_answer=sample["expected_answer"],
                nodes=sample["nodes"],
            )
            eval_samples.append(eval_sample)
        return eval_samples

    def save_as_excel(self, results: Dict[str, list], path: str = "eval_results.xlsx"):
        concat_results = []
        for i in results.values():
            concat_results += i
        df = pd.DataFrame(concat_results)
        df.to_excel(path)


if __name__ == "__main__":
    evaluators = [SetEvaluator()]
    scores = []
    evaluation = Evaluation(evaluators=evaluators, scores=scores)
    results = evaluation.run(save_as_excel=True, cache_path="cached_results/chain_results.pickle")
    print(results)
