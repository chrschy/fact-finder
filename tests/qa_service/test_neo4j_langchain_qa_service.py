import os
from typing import Any, Dict, List
from unittest.mock import ANY, MagicMock

import pytest
from dotenv import load_dotenv
from langchain.chains.graph_qa.cypher import construct_schema
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

from fact_finder.prompt_templates import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from fact_finder.qa_service.cypher_preprocessors.format_preprocessor import FormatPreprocessor
from fact_finder.qa_service.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)
from fact_finder.qa_service.cypher_preprocessors.extract_subgraph_with_llm_preprocessor import (
    ReturnSubgraphWithLLMPreprocessor,
)
from fact_finder.qa_service.cypher_preprocessors.synonym_cypher_query_preprocessor import SynonymCypherQueryPreprocessor
from fact_finder.qa_service.neo4j_langchain_qa_service import Neo4JLangchainQAService
from fact_finder.synonym_finder.synonym_finder import WikiDataSynonymFinder

load_dotenv()


def test_cypher_generation_is_called_with_expected_arguments(query, chain, cypher_llm, cypher_prompt):
    chain.invoke(query)
    cypher_llm.generate_prompt.assert_called_once_with([cypher_prompt], None, callbacks=ANY)


def test_graph_is_called_with_expected_caypher_query(query, chain, graph, cypher_query):
    chain.invoke(query)
    graph.query.assert_called_once_with(cypher_query)


def test_qa_is_called_with_expected_arguments(query, chain, qa_llm, qa_prompt):
    chain.invoke(query)
    qa_llm.generate_prompt.assert_called_once_with([qa_prompt], None, callbacks=ANY)


def test_returned_result_matches_model_output(query, chain, system_answer):
    assert chain.invoke(query)["result"] == system_answer


@pytest.mark.parametrize("cypher_query", ["SCHEMA_ERROR: This is not a cypher query!"], indirect=True)
def test_invalid_cypher_query_is_returned_directly(query, chain, cypher_query):
    assert chain.invoke(query)["result"] == cypher_query


@pytest.fixture
def query() -> str:
    return "<this is the user query>"


@pytest.fixture
def chain(
    cypher_prompt_template,
    qa_prompt_template,
    cypher_llm,
    qa_llm,
    graph,
):
    return Neo4JLangchainQAService.from_llm(
        None,
        graph=graph,
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        cypher_prompt=cypher_prompt_template,
        qa_prompt=qa_prompt_template,
        verbose=False,
        return_intermediate_steps=False,
    )


@pytest.fixture
def cypher_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["schema", "question"], template="Generate some cypher with schema:\n{schema}\nFor {question}:"
    )


@pytest.fixture
def structured_schema() -> Dict[str, Any]:
    return {
        "node_props": {
            "disease": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
            "anatomy": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
        },
        "rel_props": {},
        "relationships": [
            {"start": "disease", "type": "associated_with", "end": "gene_or_protein"},
            {"start": "disease", "type": "phenotype_present", "end": "effect_or_phenotype"},
            {"start": "disease", "type": "phenotype_absent", "end": "effect_or_phenotype"},
            {"start": "disease", "type": "parent_child", "end": "disease"},
        ],
    }


@pytest.fixture
def schema(structured_schema) -> str:
    return construct_schema(structured_schema, [], [])


@pytest.fixture
def cypher_prompt(cypher_prompt_template, schema, query) -> str:
    return cypher_prompt_template.format_prompt(schema=schema, question=query)


@pytest.fixture(params=["<this is a cypher query>"])
def cypher_query(request) -> str:
    return request.param


@pytest.fixture
def query_response() -> List[str]:
    return ["<this is the result from the graph>"]


@pytest.fixture
def qa_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"], template="Generate an answer to {question} using {context}:"
    )


@pytest.fixture
def qa_prompt(qa_prompt_template, query_response, query) -> str:
    return qa_prompt_template.format_prompt(context=query_response, question=query)


@pytest.fixture
def system_answer() -> str:
    return "<this is the answer>"


@pytest.fixture
def cypher_llm(cypher_query) -> BaseLanguageModel:
    return build_llm_mock(cypher_query)


@pytest.fixture
def qa_llm(system_answer) -> BaseLanguageModel:
    return build_llm_mock(system_answer)


@pytest.fixture
def graph(structured_schema: Dict[str, Any], query_response: List[str]) -> Neo4jGraph:
    graph = MagicMock(spec=Neo4jGraph)
    graph.get_structured_schema = structured_schema
    graph.query = MagicMock()
    graph.query.return_value = query_response
    return graph


def build_llm_mock(output: str) -> BaseLanguageModel:
    llm = MagicMock(spec=BaseLanguageModel)
    llm.generate_prompt = MagicMock()
    llm.generate_prompt.return_value = LLMResult(generations=[[Generation(text=output)]])
    return llm


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
    synonym_preprocessor = SynonymCypherQueryPreprocessor(graph=graph_e2e, synonym_finder=WikiDataSynonymFinder())
    cypher_query_formatting_preprocessor = FormatPreprocessor()
    return_all_nodes_preprocessor = ReturnSubgraphWithLLMPreprocessor(model=model_e2e)
    return [
        lower_case_preprocessor,
        synonym_preprocessor,
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
