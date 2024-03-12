from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from fact_finder.chains.graph_qa_chain import GraphQAChain
from langchain.chains.graph_qa.cypher import construct_schema
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.prompt import PromptTemplate


def test_cypher_generation_is_called_with_expected_arguments(question, chain, llm, cypher_prompt):
    chain.invoke(question)
    all_params = [p.args[0][0] for p in llm.generate_prompt.mock_calls]
    assert cypher_prompt in all_params


def test_graph_is_called_with_expected_cypher_query(question, chain, graph, cypher_query):
    chain.invoke(question)
    all_params = [p.args[0] for p in graph.query.mock_calls]
    assert cypher_query in all_params


def test_qa_is_called_with_expected_arguments(question, chain, llm, answer_prompt):
    chain.invoke(question)
    all_params = [p.args[0][0] for p in llm.generate_prompt.mock_calls]
    assert answer_prompt in all_params


def test_returned_result_matches_model_output(question, chain, system_answer):
    assert chain.invoke(question)["graph_qa_output"].answer == system_answer


@pytest.mark.parametrize("cypher_query", ["SCHEMA_ERROR: This is not a cypher query!"], indirect=True)
def test_invalid_cypher_query_is_returned_directly(question, chain, cypher_query):
    assert chain.invoke(question)["graph_qa_output"].answer == cypher_query[len("SCHEMA_ERROR: ") :]


@pytest.fixture
def chain(
    cypher_prompt_template,
    answer_generation_prompt_template,
    llm,
    graph,
):
    return GraphQAChain(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt_template,
        answer_generation_prompt=answer_generation_prompt_template,
        cypher_query_preprocessors=[],
        return_intermediate_steps=True,
    )


@pytest.fixture
def cypher_prompt(cypher_prompt_template, schema, question) -> str:
    return cypher_prompt_template.format_prompt(schema=schema, question=question)


@pytest.fixture
def cypher_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["schema", "question"], template="Generate some cypher with schema:\n{schema}\nFor {question}:"
    )


@pytest.fixture
def answer_prompt(answer_generation_prompt_template, query_response, question) -> str:
    return answer_generation_prompt_template.format_prompt(context=query_response, question=question)


@pytest.fixture
def answer_generation_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"], template="Generate an answer to {question} using {context}:"
    )


@pytest.fixture
def question() -> str:
    return "<this is the user question>"


@pytest.fixture
def llm(cypher_query: str, system_answer: str) -> BaseLanguageModel:
    def llm_side_effect(prompts: List[PromptValue], *args, **kwargs) -> LLMResult:
        if "cypher" in prompts[0].to_string().lower():
            text = cypher_query
        else:
            text = system_answer
        return LLMResult(generations=[[Generation(text=text)]])

    llm = MagicMock(spec=BaseLanguageModel)
    llm.generate_prompt = MagicMock()
    llm.generate_prompt.side_effect = llm_side_effect
    return llm


@pytest.fixture(params=["<this is a cypher query>"])
def cypher_query(request) -> str:
    return request.param


@pytest.fixture
def system_answer() -> str:
    return "<this is the answer>"


@pytest.fixture
def graph(structured_schema: Dict[str, Any], query_response: List[str]) -> Neo4jGraph:
    graph = MagicMock(spec=Neo4jGraph)
    graph.get_structured_schema = structured_schema
    graph.query = MagicMock()
    graph.query.return_value = query_response
    return graph


@pytest.fixture
def schema(structured_schema) -> str:
    return construct_schema(structured_schema, [], [])


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
def query_response() -> List[Dict[str, str]]:
    return [{"node": "<this is the result from the graph>"}]
