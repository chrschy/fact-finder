from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from langchain_community.graphs import Neo4jGraph

from fact_finder.chains.graph_chain import GraphChain


def test_graph_chain_returns_graph_result(
    graph_chain: GraphChain, preprocessors_chain_result: Dict[str, Any], graph_result: List[Dict[str, Any]]
):
    result = graph_chain(inputs=preprocessors_chain_result)
    assert result["graph_result"] == graph_result


@pytest.mark.parametrize("top_k", (0, 5, 20, 50), indirect=True)
def test_graph_result_length_is_at_most_k(
    graph_chain: GraphChain, top_k: int, preprocessors_chain_result: Dict[str, Any], graph_result: List[Dict[str, Any]]
):
    result = graph_chain(inputs=preprocessors_chain_result)
    if top_k <= len(graph_result):
        assert len(result["graph_result"]) == top_k
    else:
        assert len(result["graph_result"]) <= top_k


def test_graph_result_is_added_to_intermediate_steps(
    graph_chain: GraphChain, preprocessors_chain_result: Dict[str, Any]
):
    result = graph_chain(inputs=preprocessors_chain_result)
    assert "graph_result" in result["intermediate_steps"][-1].keys()


def test_graph_is_called_with_expected_cypher_query(graph_chain: GraphChain, graph: MagicMock):
    cypher_query = "<this is a cypher query>"
    graph_chain.invoke(input={graph_chain.input_key: cypher_query})
    graph.query.assert_called_once_with(cypher_query)


@pytest.fixture
def graph_chain(graph: Neo4jGraph, top_k: int) -> GraphChain:
    return GraphChain(graph=graph, top_k=top_k, return_intermediate_steps=True)


@pytest.fixture
def top_k(request) -> int:
    if hasattr(request, "param") and request.param:
        return request.param
    return 20


@pytest.fixture
def graph(graph_result) -> Neo4jGraph:
    graph = MagicMock(spec=Neo4jGraph)
    graph.query = MagicMock()
    graph.query.return_value = graph_result
    return graph


@pytest.fixture()
def graph_result() -> List[Dict[str, Any]]:
    return [
        {"d.name": "phenytoin"},
        {"d.name": "phenytoin"},
        {"d.name": "phenytoin"},
        {"d.name": "valproic acid"},
        {"d.name": "lamotrigine"},
        {"d.name": "lamotrigine"},
        {"d.name": "diazepam"},
        {"d.name": "clonazepam"},
        {"d.name": "fosphenytoin"},
        {"d.name": "mephenytoin"},
        {"d.name": "mephenytoin"},
        {"d.name": "neocitrullamon"},
        {"d.name": "carbamazepine"},
        {"d.name": "carbamazepine"},
        {"d.name": "phenobarbital"},
        {"d.name": "phenobarbital"},
        {"d.name": "secobarbital"},
        {"d.name": "primidone"},
        {"d.name": "primidone"},
        {"d.name": "lorazepam"},
    ]


@pytest.fixture
def preprocessors_chain_result() -> Dict[str, Any]:
    return {
        "cypher_query": "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name",
        "intermediate_steps": [
            {"question": "Which drugs are associated with epilepsy?"},
            {
                "FormatPreprocessor": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
                "LowerCasePropertiesCypherQueryPreprocessor": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
            },
        ],
        "preprocessed_cypher_query": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
    }
