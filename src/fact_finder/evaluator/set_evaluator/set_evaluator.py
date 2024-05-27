from abc import ABC
from typing import Any, Collection, Dict, Iterable, List, Optional, Set, Tuple

from fact_finder.chains.graph_qa_chain.output import GraphQAChainOutput
from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.utils import build_neo4j_graph
from langchain_community.graphs import Neo4jGraph
from tqdm import tqdm


class SetEvaluator:
    CYPHER_QUERY_TEMPLATE: str = "MATCH (n) WHERE n.index ={idx} RETURN n.name"

    def __init__(self, graph: Optional[Neo4jGraph] = None):
        self.graph = graph or build_neo4j_graph()

    def evaluate(
        self, evaluation_samples: List[EvaluationSample], chain_results: List[Dict[str, Any]], **kwargs
    ) -> List[Dict[str, Any]]:
        eval_results = []
        for sample, result in tqdm(zip(evaluation_samples, chain_results)):
            score, precision, recall = self.evaluate_sample(sample=sample, chain_result=result)
            eval_result = {
                "question": sample.question,
                "expected_cypher": sample.cypher_query,
                "expected_graph_response": sample.nodes,
                "expected_answer": sample.expected_answer,
                "score": score,
                "precision": precision,
                "recall": recall,
            }
            try:
                eval_result["actual_cypher"] = result["graph_qa_output"].cypher_query
                eval_result["actual_graph_response"] = result["graph_qa_output"].graph_response
                eval_result["actual_answer"] = result["graph_qa_output"].answer
            except KeyError:
                pass
            eval_results.append(eval_result)
        return eval_results

    def evaluate_sample(
        self, sample: EvaluationSample, chain_result: Dict[str, GraphQAChainOutput | Any]
    ) -> Tuple[float, float, float]:
        try:
            if not chain_result or not chain_result["graph_qa_output"].graph_response:
                print("No chain result or no nodes.")
                return 0.0, 0.0, 0.0
            if "index" in sample.nodes[0]:
                return self.evaluate_sample_with_single_index(sample=sample, result=chain_result)
            if "value" in sample.nodes[0]:
                return self.evaluate_sample_with_value_result(sample=sample, result=chain_result)
            return self.evaluate_sample_with_tuple_in_label(sample=sample, result=chain_result)
        except:
            print("Error")
            return 0.0, 0.0, 0.0

    def evaluate_sample_with_single_index(
        self, sample: EvaluationSample, result: Dict[str, GraphQAChainOutput | Any]
    ) -> Tuple[float, float, float]:
        ids = [node["index"] for node in sample.nodes]
        names = set(self._query_node_names(ids))
        all_scores = self._get_scores_per_key(names, result)
        return max(all_scores.values(), key=lambda s: s[0])

    def evaluate_sample_with_value_result(
        self, sample: EvaluationSample, result: Dict[str, GraphQAChainOutput | Any]
    ) -> Tuple[float, float, float]:
        value = [node["value"] for node in sample.nodes]
        all_scores = self._get_scores_per_key(value, result)
        return max(all_scores.values(), key=lambda s: s[0])

    def evaluate_sample_with_tuple_in_label(
        self, sample: EvaluationSample, result: Dict[str, GraphQAChainOutput | Any]
    ) -> Tuple[float, float, float]:
        tuple_size = max(int(key[len("index") :]) for key in sample.nodes[0] if key.startswith("index")) + 1
        indices = [f"index{i}" for i in range(tuple_size)]
        ids_per_index = {i: tuple(self._query_node_names(node[i] for node in sample.nodes)) for i in indices}
        best_graph_result_keys = {
            k: max(spk := self._get_scores_per_key(v, result), key=lambda k: spk[k][0])
            for k, v in ids_per_index.items()
        }
        expected_tuples = set(zip(*(ids_per_index[i] for i in indices)))
        result_graph_output = result["graph_qa_output"].graph_response
        result_tuples = set(tuple(res[best_graph_result_keys[i]] for i in indices) for res in result_graph_output)
        return (
            intersection_over_union(result_tuples, expected_tuples),
            precision(result_tuples, expected_tuples),
            recall(result_tuples, expected_tuples),
        )

    def _query_node_names(self, ids: Iterable[int]) -> Iterable[str]:
        for number in ids:
            graph_return = self.graph.query(self.CYPHER_QUERY_TEMPLATE.replace("{idx}", f"{number}"))
            yield graph_return[0]["n.name"]

    def _get_scores_per_key(
        self, expected_values: Collection[str], result: Dict[str, GraphQAChainOutput | Any]
    ) -> Dict[str, Tuple[float, float, float]]:
        result_graph_output: List[Dict[str, Any]] = result["graph_qa_output"].graph_response
        if len(result_graph_output) == 0:
            return {"INVALID_KEY": 0.0}
        result_graph_output_keys = list(result_graph_output[0].keys())
        return {
            key: self._compute_score_for_key(expected_values, result_graph_output, key)
            for key in result_graph_output_keys
        }

    def _compute_score_for_key(
        self, expected_values: Collection[str], result_graph_output: List[Dict[str, Any]], key: str
    ) -> Tuple[float, float, float]:
        if len(result_graph_output) == 0:
            return 0.0, 0.0, 0.0
        if isinstance(result_graph_output[0][key], dict) and "name" in result_graph_output[0][key]:
            # Handle the query returning whole nodes instead of specific properties.
            # TODO Other properties than "name"?
            entries = {i[key]["name"] for i in result_graph_output}
        else:
            entries = {i[key] for i in result_graph_output}
        return (
            intersection_over_union(entries, expected_values),
            precision(entries, expected_values),
            recall(entries, expected_values),
        )


def intersection_over_union(result: Set[Any], expected: Collection[Any]) -> float:
    return len(result.intersection(expected)) / len(result.union(expected))


def precision(result: Set[Any], expected: Collection[Any]) -> float:
    return len(result.intersection(expected)) / len(result)


def recall(result: Set[Any], expected: Collection[Any]) -> float:
    return len(result.intersection(expected)) / len(expected)
