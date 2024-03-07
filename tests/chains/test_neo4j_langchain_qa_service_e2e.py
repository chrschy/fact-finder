import os

import pytest
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

from fact_finder.chains.neo4j_langchain_qa_service import Neo4JLangchainQAService
from fact_finder.prompt_templates import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from fact_finder.tools.cypher_preprocessors.format_preprocessor import FormatPreprocessor
from fact_finder.tools.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)
from fact_finder.tools.sub_graph_extractor import LLMSubGraphExtractor

load_dotenv()


@pytest.fixture(scope="module")
def neo4j_url():
    return os.getenv("NEO4J_URL", "bolt://localhost:7687")


@pytest.fixture(scope="module")
def neo4j_user():
    return os.getenv("NEO4J_USER", "neo4j")


@pytest.fixture(scope="module")
def neo4j_pw():
    return os.getenv("NEO4J_PW", "opensesame")


@pytest.fixture(scope="module")
def open_ai_key():
    return os.getenv("OPENAI_API_KEY")


@pytest.fixture(scope="module")
def graph_e2e(neo4j_url, neo4j_user, neo4j_pw):
    return Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_pw)


@pytest.fixture(scope="module")
def model_e2e(open_ai_key):
    return ChatOpenAI(model="gpt-4", streaming=False, temperature=0, api_key=open_ai_key)


@pytest.fixture(scope="module")
def preprocessors_e2e(graph_e2e, model_e2e):
    lower_case_preprocessor = LowerCasePropertiesCypherQueryPreprocessor()
    cypher_query_formatting_preprocessor = FormatPreprocessor()
    return_all_nodes_preprocessor = LLMSubGraphExtractor(model=model_e2e)
    return [
        lower_case_preprocessor,
        cypher_query_formatting_preprocessor,
        return_all_nodes_preprocessor,
    ]


@pytest.fixture(scope="module")
def neo4j_chain_e2e(model_e2e, graph_e2e, preprocessors_e2e):
    return Neo4JLangchainQAService.from_llm(
        model_e2e,
        graph=graph_e2e,
        cypher_query_preprocessors=preprocessors_e2e,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        qa_prompt=CYPHER_QA_PROMPT,
        verbose=True,
        return_intermediate_steps=True,
    )


def run_e2e_chain(neo4j_chain_e2e: Neo4JLangchainQAService, question: str):
    message = {"query": question}
    result = neo4j_chain_e2e._call(inputs=message)
    return result


@pytest.mark.skip(reason="end to end")
def test_e2e(neo4j_chain_e2e):
    questions = [
        "Which drugs are associated with epilepsy?",
        "Which drugs are associated with schizophrenia?",
        "Which medication has the most indications?",
        "What are the phenotypes associated with cardioacrofacial dysplasia?",
    ]
    for question in questions:
        result = run_e2e_chain(neo4j_chain_e2e=neo4j_chain_e2e, question=question)
        assert len(result) == 3
        assert "sub_graph" in result.keys()
