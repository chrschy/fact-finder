import os.path
from typing import Any, Dict, List

import pandas as pd
from langchain_openai import ChatOpenAI
from tqdm import tqdm

import fact_finder.config.primekg_config as graph_config
from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluation_samples import manual_samples
from fact_finder.evaluator.set_evaluator.set_evaluator import SetEvaluator
from fact_finder.evaluator.util import load_pickle, save_pickle


class Evaluation:

    def __init__(
        self,
        run_name: str,
        model_name: str,
        chain_args: List[str] = None,
        limit_of_samples: int = None,
        idx_list_of_samples: List[int] = None,
        run_without_preprocessors: bool = False,
    ):
        if chain_args is None:
            chain_args = [
                "--skip_subgraph_generation",
                "--normalized_graph",
                "--use_entity_detection_preprocessing",
            ]
        self.model_name = model_name
        self.run_name = run_name
        self.idx_list_of_samples = idx_list_of_samples
        if run_without_preprocessors:
            build_function = graph_config.build_chain_without_preprocessings_etc
        else:
            build_function = graph_config.build_chain
        self.chain = build_function(model=self.load_model(), combine_output_with_sematic_scholar=False, args=chain_args)
        self.eval_samples = self.eval_samples(limit_of_samples=limit_of_samples)

    def run(self, save_as_excel: bool = False):
        cache_path = "cached_results/" + self.run_name + ".pickle"
        chain_results = self.run_chain(cache_path)
        results = self.evaluate(chain_results)
        if save_as_excel:
            self.save_as_excel(results)
        return results

    def evaluate(self, chain_results: List) -> Dict[str, Any]:
        print("Evaluating...")
        evaluator = SetEvaluator()
        evaluation = evaluator.evaluate(evaluation_samples=self.eval_samples, chain_results=chain_results)
        return {"set_evaluator": evaluation}

    def run_chain(self, cache_path: str):
        results = []
        print("Running Chains...")
        if os.path.exists(cache_path):
            return load_pickle(cache_path)
        eval_samples = self.eval_samples
        for eval_sample in tqdm(eval_samples):
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

    def save_as_excel(self, results: Dict[str, list]):
        concat_results = []
        for i in results.values():
            concat_results += i
        df = pd.DataFrame(concat_results)
        path = self.run_name + ".xlsx"
        df.to_excel(path)

    def load_model(self):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(model=self.model_name, streaming=False, temperature=0, api_key=OPENAI_API_KEY)


if __name__ == "__main__":
    models = ["gpt-4o", "gpt-4-turbo"]
    flags = [
        [
            "--skip_subgraph_generation",
            "--normalized_graph",
        ],
    ]
    for model in models:
        print(model)
        for flag in flags:
            print(flag)
            run_name = model + "_".join(flag)
            evaluation = Evaluation(
                run_name=run_name, chain_args=flag, model_name=model, run_without_preprocessors=False
            )
            results = evaluation.run(save_as_excel=True)
            print(run_name)
