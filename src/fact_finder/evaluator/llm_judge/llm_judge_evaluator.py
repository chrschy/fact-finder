import json
import os.path
from typing import Any, Dict, List

import pandas as pd
from langchain.chains.base import Chain
from langchain.evaluation import load_evaluator
from langchain_core.language_models import BaseChatModel
from tqdm import tqdm

import fact_finder.config.primekg_config as graph_config
import fact_finder.config.simple_config as llm_config
from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluation_samples import manual_samples
from fact_finder.evaluator.set_evaluator.set_evaluator import SetEvaluator
from fact_finder.evaluator.util import load_pickle, save_pickle
from fact_finder.prompt_templates import (
    LLM_JUDGE_CRITERIA_COMPLETENESS,
    LLM_JUDGE_CRITERIA_CORRECTNESS,
    LLM_JUDGE_PAIRWISE_PROMPT,
    LLM_JUDGE_SCORE_PROMPT,
)
from fact_finder.utils import load_chat_model


class LlmJudgeEvaluator:

    def __init__(
        self,
        chat_model: BaseChatModel = None,
        llm_chain: Chain = None,
        graph_chain: Chain = None,
        llm_chain_args: List[str] = [],
        graph_chain_args: List[str] = [
            "--skip_subgraph_generation",
            "--normalized_graph",
            "--use_entity_detection_preprocessing",
        ],
        limit_of_samples: int = None,
        idx_list_of_samples: List[int] = None,
    ):
        self.idx_list_of_samples = idx_list_of_samples
        if not chat_model:
            self.chat_model = load_chat_model()
        if not llm_chain:
            self.llm_chain = llm_config.build_chain(model=self.chat_model, args=llm_chain_args)
        if not graph_chain:
            self.graph_chain = graph_config.build_chain(
                model=self.chat_model, combine_output_with_sematic_scholar=False, args=graph_chain_args
            )
        self.eval_samples = self.eval_samples(limit_of_samples=limit_of_samples)

        self.evaluator_pairwise_correct = load_evaluator(
            "labeled_pairwise_string",
            llm=self.chat_model,
            prompt=LLM_JUDGE_PAIRWISE_PROMPT,
            criteria=LLM_JUDGE_CRITERIA_CORRECTNESS,
        )
        self.evaluator_pairwise_complete = load_evaluator(
            "labeled_pairwise_string",
            llm=self.chat_model,
            prompt=LLM_JUDGE_PAIRWISE_PROMPT,
            criteria=LLM_JUDGE_CRITERIA_COMPLETENESS,
        )
        self.evaluator_score_correct = load_evaluator(
            "labeled_score_string",
            llm=self.chat_model,
            prompt=LLM_JUDGE_SCORE_PROMPT,
            criteria=LLM_JUDGE_CRITERIA_CORRECTNESS,
        )
        self.evaluator_score_complete = load_evaluator(
            "labeled_score_string",
            llm=self.chat_model,
            prompt=LLM_JUDGE_SCORE_PROMPT,
            criteria=LLM_JUDGE_CRITERIA_COMPLETENESS,
        )

    def run(
        self,
        save_as_excel: bool = False,
        llm_cache_path="cached_results/llm_chain_results.pickle",
        graph_cache_path="cached_results/graph_chain_results.pickle",
    ):
        llm_chain_results = self.run_chain(self.llm_chain, llm_cache_path)
        graph_chain_results = self.run_chain(self.graph_chain, graph_cache_path)
        results = self.evaluate(llm_chain_results, graph_chain_results)
        if save_as_excel:
            self.save_as_excel(results)
        return results

    def evaluate(self, llm_chain_results, graph_chain_results) -> Dict[str, Any]:
        print("Evaluating...")
        eval_results = []
        for sample, llm_result, graph_result in tqdm(zip(self.eval_samples, llm_chain_results, graph_chain_results)):
            cypher_response = (
                graph_result["graph_qa_output"].graph_response if "graph_qa_output" in graph_result else ""
            )
            node_names = sample.nodes
            if "index" in node_names[0]:
                node_names = list(SetEvaluator().query_node_names([node["index"] for node in node_names]))

            print("\nSAMPLE")
            llm_judge_pairwise_correct = self.evaluator_pairwise_correct.evaluate_string_pairs(
                prediction=llm_result["text"],
                prediction_b=graph_result["graph_qa_output"].answer if "graph_qa_output" in graph_result else "",
                input=sample.question,
                reference=str([i for i in node_names]),
            )

            llm_judge_pairwise_complete = self.evaluator_pairwise_complete.evaluate_string_pairs(
                prediction=llm_result["text"],
                prediction_b=graph_result["graph_qa_output"].answer if "graph_qa_output" in graph_result else "",
                input=sample.question,
                reference=str([i for i in node_names]),
            )

            llm_judge_score_correct = self.evaluator_score_correct.evaluate_strings(
                prediction=graph_result["graph_qa_output"].answer if "graph_qa_output" in graph_result else "",
                prediction_b="",
                input=sample.question,
                reference=str(cypher_response),
            )

            llm_judge_score_complete = self.evaluator_score_complete.evaluate_strings(
                prediction=graph_result["graph_qa_output"].answer if "graph_qa_output" in graph_result else "",
                prediction_b="",
                input=sample.question,
                reference=str(cypher_response),
            )

            eval_result = {
                "question": sample.question,
                "reference": str(sample.nodes),
                "reference_names": str([i for i in node_names]),
                "expected_answer": sample.expected_answer,
                "llm_answer": llm_result["text"],
                "graph_answer": graph_result["graph_qa_output"].answer if "graph_qa_output" in graph_result else "",
                "cypher_response": cypher_response,
                "pairwise_correct": llm_judge_pairwise_correct["value"].replace("A", "LLM").replace("B", "GRAPH"),
                "pairwise_complete": llm_judge_pairwise_complete["value"].replace("A", "LLM").replace("B", "GRAPH"),
                "score_correct": str(llm_judge_score_correct["score"]).replace("8", "0"),
                "score_complete": str(llm_judge_score_complete["score"]).replace("8", "0"),
                "reasoning_pairwise_correct": llm_judge_pairwise_correct["reasoning"],
                "reasoning_pairwise_complete": llm_judge_pairwise_complete["reasoning"],
                "reasoning_score_correct": llm_judge_score_correct["reasoning"],
                "reasoning_score_complete": llm_judge_score_complete["reasoning"],
            }
            eval_results.append(eval_result)

        return {"PairwiseEvaluator": eval_results}

    def run_chain(self, chain: Chain, cache_path: str):
        results = []
        print("Running Chain...")
        if os.path.exists(cache_path):
            return load_pickle(cache_path)
        for eval_sample in tqdm(self.eval_samples):
            inputs = {"question": eval_sample.question}
            try:
                result = chain.invoke(inputs)
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
                cypher_query=sample["expected_cypher"] if "expected_cypher" in sample else "",
                expected_answer=sample["expected_answer"],
                nodes=sample["nodes"],
            )
            eval_samples.append(eval_sample)
        return eval_samples

    def save_as_excel(self, results: Dict[str, list], path: str = "eval_results_llmjudge.xlsx"):
        concat_results = []
        for i in results.values():
            concat_results += i
        df = pd.DataFrame(concat_results)
        df.to_excel(path)


if __name__ == "__main__":
    evaluation = LlmJudgeEvaluator()
    results = evaluation.run(save_as_excel=True)
