from unittest.mock import MagicMock

import pytest
from langchain_community.graphs import Neo4jGraph

from fact_finder.chains.graph_chain import GraphChain


def test_graph_chain(graph, return_intermediate_steps, preprocessors_chain_result, expected_result):
    k = 20
    chain = GraphChain(graph=graph, return_intermediate_steps=return_intermediate_steps, top_k=k)
    result = chain(inputs=preprocessors_chain_result)
    assert result["graph_result"] == expected_result
    assert len(result["graph_result"]) == k
    assert "graph_result" in result["intermediate_steps"][1].keys()


@pytest.fixture
def graph(expected_result):
    graph = MagicMock(spec=Neo4jGraph)
    graph.query = MagicMock()
    graph.query.return_value = expected_result
    return graph


@pytest.fixture
def return_intermediate_steps():
    return True


@pytest.fixture
def preprocessors_chain_result():
    return {
        "cypher_query": "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name",
        "intermediate_steps": [
            {
                "FormatPreprocessor": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
                "LowerCasePropertiesCypherQueryPreprocessor": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
            }
        ],
        "preprocessed_cypher_query": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
    }


@pytest.fixture()
def expected_result():
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
