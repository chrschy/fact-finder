import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from fact_finder.qa_service.cypher_preprocessors.extract_subgraph_with_llm_preprocessor import (
    ReturnSubgraphWithLLMPreprocessor,
)

load_dotenv()


@pytest.fixture
def open_ai_key():
    return os.getenv("OPENAI_API_KEY")


@pytest.fixture
def llm_e2e(open_ai_key):
    return ChatOpenAI(model="gpt-4", streaming=False, temperature=0, api_key=open_ai_key)


@pytest.fixture
def preprocesor(llm_e2e):
    return ReturnSubgraphWithLLMPreprocessor(model=llm_e2e)


def test_preprocessor_e2e(preprocesor):
    cypher_queries = [
        (
            """MATCH (d:disease)-[:phenotype_present]-({name:"eczema"})
            MATCH (d)-[:phenotype_present]-({name:"neutropenia"})
            MATCH (d)-[:phenotype_present]-({name:"high forehead"})
            RETURN DISTINCT d.name""",
            """MATCH (d:disease)-[r1:phenotype_present]-({name:"eczema"})
MATCH (d)-[r2:phenotype_present]-({name:"neutropenia"})
MATCH (d)-[r3:phenotype_present]-({name:"high forehead"})
RETURN DISTINCT d, r1, r2, r3""",
        ),
        (
            "MATCH (d:disease {name: 'schizophrenia'})-[:indication]->(g:drug) RETURN g",
            "MATCH (d:disease {name: 'schizophrenia'})-[r:indication]->(g:drug) RETURN r, d, g",
        ),
        (
            "MATCH (d:disease {name: 'epilepsy'})-[:indication]->(g:drug) RETURN g",
            "MATCH (d:disease {name: 'epilepsy'})-[r:indication]->(g:drug) RETURN d, r, g",
        ),
        (
            "MATCH (d:disease {name: 'epilepsy'})-[:indication]->(g:drug) RETURN g.name",
            "MATCH (d:disease {name: 'epilepsy'})-[r:indication]->(g:drug) RETURN d, r, g",
        ),
    ]
    for cypher_query, expected_result in cypher_queries:
        result = preprocesor(cypher_query)
        assert result == expected_result
