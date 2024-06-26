import os
from unittest.mock import patch

import pytest
from dotenv import dotenv_values
from fact_finder.tools.sub_graph_extractor import LLMSubGraphExtractor
from langchain_openai import ChatOpenAI


@pytest.mark.skip(reason="end to end")
@patch.dict(os.environ, {**dotenv_values(), **os.environ})
def test_subgraph_extractor_e2e(llm_subgraph_extractor):
    assert_subgraph_extractor(
        llm_subgraph_extractor,
        "MATCH (d:disease {name: 'schizophrenia'})-[:indication]->(g:drug) RETURN g",
        [
            "MATCH (d:disease {name: 'schizophrenia'})-[r:indication]->(g:drug) RETURN d, r, g",
            "MATCH (d:disease {name: 'schizophrenia'})-[r:indication]->(g:drug) RETURN r, d, g",
        ],
    )
    assert_subgraph_extractor(
        llm_subgraph_extractor,
        "MATCH (d:disease {name: 'epilepsy'})-[:indication]->(g:drug) RETURN g",
        [
            "MATCH (d:disease {name: 'epilepsy'})-[r:indication]->(g:drug) RETURN r, d, g",
            "MATCH (d:disease {name: 'epilepsy'})-[r:indication]->(g:drug) RETURN d, r, g",
        ],
    )
    assert_subgraph_extractor(
        llm_subgraph_extractor,
        "MATCH (d:disease {name: 'epilepsy'})-[:indication]->(g:drug) RETURN g.name",
        [
            "MATCH (d:disease {name: 'epilepsy'})-[r:indication]->(g:drug) RETURN r, d, g",
            "MATCH (d:disease {name: 'epilepsy'})-[r:indication]->(g:drug) RETURN d, r, g",
        ],
    )


@pytest.mark.skip(reason="end to end")
@patch.dict(os.environ, {**dotenv_values(), **os.environ})
def test_more_complicated_query_subgraph_extractor_e2e(llm_subgraph_extractor):
    complicated_query = """MATCH (d:disease)-[:phenotype_present]-({name:"eczema"}) MATCH (d)-[:phenotype_present]-({
    name:"neutropenia"}) MATCH (d)-[:phenotype_present]-({name:"high forehead"}) RETURN DISTINCT d.name"""
    result = llm_subgraph_extractor(complicated_query)
    assert len(result) > len(complicated_query)


def assert_subgraph_extractor(
    llm_subgraph_extractor: LLMSubGraphExtractor, cypher_query: str, expected_results: list[str]
):
    result = llm_subgraph_extractor(cypher_query)
    assert result in expected_results


@pytest.fixture
def llm_subgraph_extractor(llm_e2e):
    return LLMSubGraphExtractor(model=llm_e2e)


@pytest.fixture
def llm_e2e(open_ai_key):
    return ChatOpenAI(model="gpt-4", streaming=False, temperature=0, api_key=open_ai_key)


@pytest.fixture
def open_ai_key():
    return os.getenv("OPENAI_API_KEY")
