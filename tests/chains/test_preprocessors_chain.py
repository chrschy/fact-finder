from unittest.mock import MagicMock

import pytest
from langchain_community.graphs import Neo4jGraph

from fact_finder.chains.cypher_preprocessors.format_preprocessor import FormatPreprocessor
from fact_finder.chains.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)
from fact_finder.chains.cypher_query_preprocessors_chain import CypherQueryPreprocessorsChain


def test_preprocessor_chain(cypher_query_generation_chain_result, preprocessors):
    chain = CypherQueryPreprocessorsChain(return_intermediate_steps=True, cypher_query_preprocessors=preprocessors)
    result = chain(cypher_query_generation_chain_result)
    assert chain.output_key in result.keys()
    assert len(result["intermediate_steps"][1]) == len(preprocessors)
    assert (
        result["preprocessed_cypher_query"]
        == 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name'
    )
    assert cypher_query_generation_chain_result["cypher_query"] != result["preprocessed_cypher_query"]


@pytest.fixture
def cypher_query_generation_chain_result():
    return {
        "cypher_query": "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name",
        "intermediate_steps": [{"question": "Which drugs are associated with epilepsy?"}],
    }


@pytest.fixture
def preprocessors(graph):
    return [FormatPreprocessor(), LowerCasePropertiesCypherQueryPreprocessor()]


@pytest.fixture
def graph():
    return MagicMock(spec=Neo4jGraph)
