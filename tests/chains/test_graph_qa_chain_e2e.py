import os
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from dotenv import dotenv_values
from fact_finder.chains.graph_qa_chain import GraphQAChain
from fact_finder.prompt_templates import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import (
    CypherQueryPreprocessor,
)
from fact_finder.tools.cypher_preprocessors.format_preprocessor import (
    FormatPreprocessor,
)
from fact_finder.tools.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI


@pytest.mark.skip(reason="end to end")
@patch.dict(os.environ, {**dotenv_values(), **os.environ})
def test_e2e(e2e_chain: GraphQAChain):
    questions = [
        "Which drugs are associated with epilepsy?",
        "Which drugs are associated with schizophrenia?",
        "Which medication has the most indications?",
        "What are the phenotypes associated with cardioacrofacial dysplasia?",
    ]
    for question in questions:
        result = run_e2e_chain(e2e_chain=e2e_chain, question=question)
        assert len(result) == 3
        assert e2e_chain.output_key in result.keys() and e2e_chain.intermediate_steps_key in result.keys()


def run_e2e_chain(e2e_chain: GraphQAChain, question: str) -> Dict[str, Any]:
    message = {e2e_chain.input_key: question}
    return e2e_chain.invoke(input=message)


@pytest.fixture(scope="module")
def e2e_chain(model_e2e: ChatOpenAI, graph_e2e: Neo4jGraph, preprocessors_e2e: list) -> GraphQAChain:
    return GraphQAChain(
        llm=model_e2e,
        graph=graph_e2e,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        answer_generation_prompt=CYPHER_QA_PROMPT,
        cypher_query_preprocessors=preprocessors_e2e,
        return_intermediate_steps=True,
    )


@pytest.fixture(scope="module")
def preprocessors_e2e() -> List[CypherQueryPreprocessor]:
    cypher_query_formatting_preprocessor = FormatPreprocessor()
    lower_case_preprocessor = LowerCasePropertiesCypherQueryPreprocessor()
    return [cypher_query_formatting_preprocessor, lower_case_preprocessor]


@pytest.fixture(scope="module")
def graph_e2e(neo4j_url: str, neo4j_user: str, neo4j_pw: str) -> Neo4jGraph:
    return Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_pw)


@pytest.fixture(scope="module")
def model_e2e(open_ai_key: str) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4", streaming=False, temperature=0, api_key=open_ai_key)


@pytest.fixture(scope="module")
def neo4j_url() -> str:
    return os.getenv("NEO4J_URL", "bolt://localhost:7687")


@pytest.fixture(scope="module")
def neo4j_user() -> str:
    return os.getenv("NEO4J_USER", "neo4j")


@pytest.fixture(scope="module")
def neo4j_pw() -> str:
    return os.getenv("NEO4J_PW", "opensesame")


@pytest.fixture(scope="module")
def open_ai_key() -> str:
    return os.getenv("OPENAI_API_KEY")
