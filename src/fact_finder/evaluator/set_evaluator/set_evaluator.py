from abc import ABC
from typing import Any, Collection, Dict, Iterable, List, Optional, Set

from fact_finder.chains.graph_qa_chain.output import GraphQAChainOutput
from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.utils import build_neo4j_graph
from langchain_community.graphs import Neo4jGraph


class SetEvaluator:
    CYPHER_QUERY_TEMPLATE: str = "MATCH (n) WHERE n.index ={idx} RETURN n.name"

    def __init__(self, graph: Optional[Neo4jGraph] = None):
        self.graph = graph or build_neo4j_graph()

    def evaluate(
        self,
        evaluation_samples: List[EvaluationSample],
        chain_results: List[Dict[str, GraphQAChainOutput | Any]],
        **kwargs,
    ) -> List[float]:
        scores = []
        for sample, result in zip(evaluation_samples, chain_results):
            scores.append(self.evaluate_sample(sample, result))
        return scores

    def evaluate_sample(self, sample: EvaluationSample, result: Dict[str, GraphQAChainOutput | Any]) -> float:
        if "index" in sample.nodes[0]:
            return self.evaluate_sample_with_single_index(sample=sample, result=result)
        elif "value" in sample.nodes[0]:
            return self.evaluate_sample_with_value_result(sample=sample, result=result)
        else:
            return self.evaluate_sample_with_tuple_in_label(sample=sample, result=result)

    def evaluate_sample_with_single_index(
        self, sample: EvaluationSample, result: Dict[str, GraphQAChainOutput | Any]
    ) -> float:
        ids = [node["index"] for node in sample.nodes]
        names = set(self._query_node_names(ids))
        all_scores = self._get_scores_per_key(names, result)
        return max(all_scores.values())

    def evaluate_sample_with_value_result(
        self, sample: EvaluationSample, result: Dict[str, GraphQAChainOutput | Any]
    ) -> float:
        value = [node["value"] for node in sample.nodes]
        all_scores = self._get_scores_per_key(value, result)
        return max(all_scores.values())

    def evaluate_sample_with_tuple_in_label(
        self, sample: EvaluationSample, result: Dict[str, GraphQAChainOutput | Any]
    ) -> float:
        tuple_size = max(int(key[len("index") :]) for key in sample.nodes[0] if key.startswith("index")) + 1
        indices = [f"index{i}" for i in range(tuple_size)]
        ids_per_index = {i: tuple(self._query_node_names(node[i] for node in sample.nodes)) for i in indices}
        best_graph_result_keys = {
            k: max(spk := self._get_scores_per_key(v, result), key=spk.get) for k, v in ids_per_index.items()
        }
        expected_tuples = set(zip(*(ids_per_index[i] for i in indices)))
        result_graph_output = result["graph_qa_output"].graph_response
        result_tuples = set(tuple(res[best_graph_result_keys[i]] for i in indices) for res in result_graph_output)
        return intersection_over_union(expected_tuples, result_tuples)

    def _query_node_names(self, ids: Iterable[int]) -> Iterable[str]:
        for number in ids:
            graph_return = self.graph.query(self.CYPHER_QUERY_TEMPLATE.replace("{idx}", f"{number}"))
            yield graph_return[0]["n.name"]

    def _get_scores_per_key(
        self, expected_values: Collection[str], result: Dict[str, GraphQAChainOutput | Any]
    ) -> Dict[str, float]:
        result_graph_output: List[Dict[str, Any]] = result["graph_qa_output"].graph_response
        if len(result_graph_output) == 0:
            return {"INVALID_KEY": 0.0}
        result_graph_output_keys = list(result_graph_output[0].keys())
        all_scores = {}
        for key in result_graph_output_keys:
            entries = {i[key] for i in result_graph_output}
            all_scores[key] = intersection_over_union(entries, expected_values)
        return all_scores


def intersection_over_union(s1: Set[Any], s2: Collection[Any]) -> float:
    return len(s1.intersection(s2)) / len(s1.union(s2))
