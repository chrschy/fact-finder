from unittest.mock import MagicMock

import pytest
from langchain_community.graphs import Neo4jGraph

from fact_finder.chains.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor
from fact_finder.chains.cypher_preprocessors.format_preprocessor import FormatPreprocessor
from fact_finder.chains.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)
from fact_finder.chains.cypher_query_preprocessors_chain import CypherQueryPreprocessorsChain
from fact_finder.config.primekg_config import _build_preprocessors


@pytest.fixture
def cypher_query_generation_chain_result():
    return {
        "cypher_query": "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name",
        "intermediate_steps": [],
    }


@pytest.fixture
def graph():
    return MagicMock(spec=Neo4jGraph)


@pytest.fixture
def preprocessors(graph):
    return [FormatPreprocessor(), LowerCasePropertiesCypherQueryPreprocessor()]


def test_preprocessor_chain_e2e(cypher_query_generation_chain_result, preprocessors):
    chain = CypherQueryPreprocessorsChain(return_intermediate_steps=True, cypher_query_preprocessors=preprocessors)
    result = chain(cypher_query_generation_chain_result)
