from typing import List, Dict, Any
from unittest.mock import MagicMock

import pytest

from fact_finder.tools.subgraph_extension import SubgraphExpansion


@pytest.fixture
def neo4j_graph():
    return MagicMock()


@pytest.fixture
def subgraph_expansion(neo4j_graph):
    return SubgraphExpansion(graph=neo4j_graph)


@pytest.fixture
def extracted_nodes() -> List[Dict[str, Any]]:
    return [
        {
            "d": {
                "index": 27937,
                "source": "mondo_grouped",
                "causes": "nutrition",
                "name": "cardioacrofacial dysplasia",
                "id": "30876_30877_31386",
            },
            "pp": (
                {
                    "index": 27937,
                    "source": "mondo_grouped",
                    "causes": "nutrition",
                    "name": "cardioacrofacial dysplasia",
                    "id": "30876_30877_31386",
                    "management_and_treatment": "no information available",
                    "prevention": "no information available",
                },
                "phenotype_present",
                {
                    "name": "mandibular prognathia",
                    "index": 84579,
                    "source": "hpo",
                },
            ),
            "p": {
                "name": "mandibular prognathia",
                "index": 84579,
                "source": "hpo",
            },
            "properties(pp)": {},
        },
    ]


@pytest.fixture
def graph_output() -> List[Dict[str, Any]]:
    return [
        {
            "a": {
                "index": 27937,
                "source": "mondo_grouped",
                "causes": "nutrition",
                "name": "cardioacrofacial dysplasia",
                "id": "30876_30877_31386",
            },
            "pp": (
                {
                    "index": 27937,
                    "source": "mondo_grouped",
                    "causes": "nutrition",
                    "name": "cardioacrofacial dysplasia",
                    "id": "30876_30877_31386",
                    "management_and_treatment": "no information available",
                    "prevention": "no information available",
                },
                "a_new_relation",
                {
                    "name": "intermediate node",
                    "index": 4711,
                    "source": "hpo",
                },
            ),
            "c": {
                "name": "intermediate node",
                "index": 4711,
                "source": "hpo",
            },
            "properties(pp)": {},
        },
        {
            "b": {
                "name": "mandibular prognathia",
                "index": 84579,
                "source": "hpo",
            },
            "pp": (
                {
                    "name": "mandibular prognathia",
                    "index": 84579,
                    "source": "hpo",
                },
                "phenotype_present",
                {
                    "name": "intermediate node",
                    "index": 4711,
                    "source": "hpo",
                },
            ),
            "c": {
                "name": "intermediate node",
                "index": 4711,
                "source": "hpo",
            },
            "properties(pp)": {},
        },
    ]


def test_expansion(subgraph_expansion, neo4j_graph, extracted_nodes, graph_output):
    neo4j_graph.query.return_value = graph_output
    enriched = subgraph_expansion.expand(nodes=extracted_nodes)
    assert len(enriched) == 3
    assert extracted_nodes[0] in enriched
    assert graph_output[0] in enriched
    assert graph_output[1] in enriched
