from abc import ABC
from typing import List, Dict, Any

from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.utils import build_neo4j_graph


class SetEvaluator:
    CYPHER_QUERY_TEMPLATE: str = "MATCH (n) WHERE n.index ={idx} RETURN n.name"

    def __init__(self):
        self.graph = build_neo4j_graph()

    def evaluate(
        self, evaluation_samples: List[EvaluationSample], chain_results: List[Dict[str, Any]], **kwargs
    ) -> List[float]:
        scores = []
        for sample, result in zip(evaluation_samples, chain_results):
            scores.append(self.evaluate_single(sample=sample, result=result))
        return scores

    def evaluate_single(self, sample: EvaluationSample, result: Dict[str, Any]) -> float:
        ids = [node["index"] for node in sample.nodes]
        names = []
        for number in ids:
            graph_return = self.graph.query(self.CYPHER_QUERY_TEMPLATE.replace("{idx}", f"{number}"))
            names.append(graph_return[0]["n.name"])
        result_graph_output = result["graph_qa_output"].graph_response
        assert result_graph_output
        result_graph_output_keys = list(result_graph_output[0].keys())
        all_scores = {}
        for key in result_graph_output_keys:
            entries = {i[key] for i in result_graph_output}
            score = len(entries.intersection(names)) / len(entries.union(names))
            all_scores[key] = score
        return max(all_scores.values())
