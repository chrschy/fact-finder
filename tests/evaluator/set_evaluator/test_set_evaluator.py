from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

import pytest
from fact_finder.chains.graph_qa_chain.output import GraphQAChainOutput
from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.set_evaluator.set_evaluator import SetEvaluator
from langchain_community.graphs import Neo4jGraph

nodes1 = [{"index": 1}, {"index": 5}, {"index": 13}]
graph_response1 = [[{"n.name": "1"}], [{"n.name": "5"}], [{"n.name": "13"}]]
sample_response1a = [{"name1": "1", "name2": "1"}, {"name1": "5", "name2": "1"}, {"name1": "13", "name2": "1"}]
sample_response1b = [{"name1": "1", "name2": "1"}, {"name1": "5", "name2": "1"}, {"name1": "12", "name2": "1"}]
sample_response1c = [{"name1": "10", "name2": "0"}, {"name1": "11", "name2": "0"}, {"name1": "12", "name2": "0"}]

nodes2 = [{"index0": 1, "index1": 2}, {"index0": 5, "index1": 6}, {"index0": 13, "index1": 14}]
graph_response2 = [
    [{"n.name": "1"}],
    [{"n.name": "5"}],
    [{"n.name": "13"}],
    [{"n.name": "2"}],
    [{"n.name": "6"}],
    [{"n.name": "14"}],
]
sample_response2a = [
    {"name1": "1", "name2": "2", "name3": "1"},
    {"name1": "5", "name2": "6", "name3": "1"},
    {"name1": "13", "name2": "14", "name3": "1"},
]
sample_response2b = [
    {"name1": "1", "name2": "2", "name3": "1"},
    {"name1": "5", "name2": "6", "name3": "1"},
    {"name1": "12", "name2": "14", "name3": "1"},
]
sample_response2c = [
    {"name1": "10", "name2": "2", "name3": "0"},
    {"name1": "11", "name2": "6", "name3": "0"},
    {"name1": "12", "name2": "14", "name3": "0"},
]

sample_response3 = [
    {"resultNode": {"name": "1", "index": 1, "id": "id1"}},
    {"resultNode": {"name": "5", "index": 5, "id": "id5"}},
    {"resultNode": {"name": "13", "index": 13, "id": "id13"}},
]


@pytest.mark.parametrize(
    "graph_result,ground_truth_nodes,result_graph_response",
    [(graph_response1, nodes1, sample_response1a)],
    indirect=True,
)
def test_evaluate_matching_sample_and_result(
    assert_evaluation_produces_correct_scores: Callable[[float, float, float], None]
):
    assert_evaluation_produces_correct_scores(iou=1.0, precision=1.0, recall=1.0)


@pytest.mark.parametrize(
    "graph_result,ground_truth_nodes,result_graph_response",
    [(graph_response1, nodes1, sample_response1b)],
    indirect=True,
)
def test_evaluate_partial_matching_sample_and_result(
    assert_evaluation_produces_correct_scores: Callable[[float, float, float], None]
):
    assert_evaluation_produces_correct_scores(iou=2 / 4, precision=2 / 3, recall=2 / 3)


@pytest.mark.parametrize(
    "graph_result,ground_truth_nodes,result_graph_response",
    [(graph_response1, nodes1, sample_response1c)],
    indirect=True,
)
def test_evaluate_not_matching_sample_and_result(
    assert_evaluation_produces_correct_scores: Callable[[float, float, float], None]
):
    assert_evaluation_produces_correct_scores(iou=0.0, precision=0.0, recall=0.0)


@pytest.mark.parametrize(
    "graph_result,ground_truth_nodes,result_graph_response",
    [(graph_response2, nodes2, sample_response2a)],
    indirect=True,
)
def test_evaluate_matching_tuple_sample_and_result(
    assert_evaluation_produces_correct_scores: Callable[[float, float, float], None]
):
    assert_evaluation_produces_correct_scores(iou=1.0, precision=1.0, recall=1.0)


@pytest.mark.parametrize(
    "graph_result,ground_truth_nodes,result_graph_response",
    [(graph_response2, nodes2, sample_response2b)],
    indirect=True,
)
def test_evaluate_partial_matching_tuple_sample_and_result(
    assert_evaluation_produces_correct_scores: Callable[[float, float, float], None]
):
    assert_evaluation_produces_correct_scores(iou=2 / 4, precision=2 / 3, recall=2 / 3)


@pytest.mark.parametrize(
    "graph_result,ground_truth_nodes,result_graph_response",
    [(graph_response2, nodes2, sample_response2c)],
    indirect=True,
)
def test_evaluate_not_matching_tuple_sample_and_result(
    assert_evaluation_produces_correct_scores: Callable[[float, float, float], None]
):
    assert_evaluation_produces_correct_scores(iou=0.0, precision=0.0, recall=0.0)


@pytest.mark.parametrize(
    "result_graph_response,ground_truth_nodes", [([{"count": 10, "other_count": 4}], [{"value": 10}])], indirect=True
)
def test_evaluate_with_matching_int_value(
    assert_evaluation_produces_correct_scores: Callable[[float, float, float], None]
):
    assert_evaluation_produces_correct_scores(iou=1.0, precision=1.0, recall=1.0)


@pytest.mark.parametrize(
    "result_graph_response,ground_truth_nodes", [([{"mean": 0.4, "var": 0.1}], [{"value": 0.4}])], indirect=True
)
def test_evaluate_with_matching_float_value(
    assert_evaluation_produces_correct_scores: Callable[[float, float, float], None]
):
    assert_evaluation_produces_correct_scores(iou=1.0, precision=1.0, recall=1.0)


@pytest.mark.parametrize(
    "result_graph_response,ground_truth_nodes", [([{"count": 11, "other_count": 4}], [{"value": 10}])], indirect=True
)
def test_evaluate_with_no_matching_int_value(
    assert_evaluation_produces_correct_scores: Callable[[float, float, float], None]
):
    assert_evaluation_produces_correct_scores(iou=0.0, precision=0.0, recall=0.0)


@pytest.mark.parametrize(
    "graph_result,ground_truth_nodes,result_graph_response",
    [(graph_response1, nodes1, sample_response3)],
    indirect=True,
)
def test_evaluate_with_whole_nodes_in_graph_response(
    assert_evaluation_produces_correct_scores: Callable[[float, float, float], None]
):
    assert_evaluation_produces_correct_scores(iou=1.0, precision=1.0, recall=1.0)


@pytest.fixture
def assert_evaluation_produces_correct_scores(
    graph: Neo4jGraph, ground_truth: EvaluationSample, result: Dict[str, GraphQAChainOutput]
) -> Callable[[float, float, float], None]:
    def run_eval(iou: float, precision: float, recall: float):
        evaluator = SetEvaluator(graph)
        eval_res = evaluator.evaluate([ground_truth], [result])[0]
        assert eval_res["score"] == iou
        assert eval_res["precision"] == precision
        assert eval_res["recall"] == recall

    return run_eval


@pytest.fixture
def graph(graph_result: List[Dict[str, Any]]) -> Neo4jGraph:
    graph = MagicMock(spec=Neo4jGraph)
    graph.query = MagicMock()
    graph.query.side_effect = graph_result
    return graph


@pytest.fixture
def result(
    ground_truth: EvaluationSample, result_graph_response: List[Dict[str, Any]]
) -> Dict[str, GraphQAChainOutput]:
    return {
        "graph_qa_output": GraphQAChainOutput(
            question=ground_truth.question,
            cypher_query=ground_truth.cypher_query,
            graph_response=result_graph_response,
            answer="",
            evidence_sub_graph=[],
            expanded_evidence_subgraph=[],
        )
    }


@pytest.fixture
def ground_truth(ground_truth_nodes: List[Dict[str, int | str]]) -> EvaluationSample:
    return EvaluationSample(
        question="How do you do?",
        cypher_query="Query the cypher!",
        question_is_answerable=True,
        nodes=ground_truth_nodes,
    )


@pytest.fixture
def ground_truth_nodes(request: pytest.FixtureRequest) -> List[Dict[str, int | str]]:
    if hasattr(request, "param") and request.param is not None:
        return request.param
    return []


@pytest.fixture
def result_graph_response(request: pytest.FixtureRequest) -> List[Dict[str, Any]]:
    if hasattr(request, "param") and request.param is not None:
        return request.param
    return []


@pytest.fixture
def graph_result(request: pytest.FixtureRequest) -> List[Dict[str, Any]]:
    if hasattr(request, "param") and request.param is not None:
        return request.param
    return []
