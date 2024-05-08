from abc import ABC
from typing import List, Dict, Any

from tqdm import tqdm

from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.utils import build_neo4j_graph


class SetEvaluator:
    CYPHER_QUERY_TEMPLATE: str = "MATCH (n) WHERE n.index ={idx} RETURN n.name"

    def __init__(self):
        self.graph = build_neo4j_graph()

    def evaluate(
        self, evaluation_samples: List[EvaluationSample], chain_results: List[Dict[str, Any]], **kwargs
    ) -> List[Dict[str, Any]]:
        eval_results = []
        for sample, result in tqdm(zip(evaluation_samples, chain_results)):
            score = self.evaluate_single(sample=sample, chain_result=result)
            eval_result = {
                "question": sample.question,
                "expected_cypher": sample.cypher_query,
                "expected_graph_response": sample.nodes,
                "expected_answer": sample.expected_answer,
                "score": score,
            }
            try:
                eval_result["actual_cypher"] = result["graph_qa_output"].cypher_query
                eval_result["actual_graph_response"] = result["graph_qa_output"].graph_response
                eval_result["actual_answer"] = result["graph_qa_output"].answer
            except KeyError:
                pass
            eval_results.append(eval_result)
        return eval_results

    def evaluate_single(self, sample: EvaluationSample, chain_result: Dict[str, Any]) -> float:
        if not chain_result:
            return 0.0
        ids = [node["index"] for node in sample.nodes]
        names = set()
        for number in ids:
            graph_query = self.CYPHER_QUERY_TEMPLATE.replace("{idx}", f"{number}")
            graph_return = self.graph.query(graph_query)
            names.add(graph_return[0]["n.name"])
        result_graph_output = chain_result["graph_qa_output"].graph_response
        if not result_graph_output:
            return 0.0
        result_graph_output_keys = list(result_graph_output[0].keys())
        all_scores = {}
        for key in result_graph_output_keys:
            entries = {i[key] for i in result_graph_output}
            score = len(entries.intersection(names)) / len(entries.union(names))
            all_scores[key] = score
        final_score = max(all_scores.values())
        return final_score
