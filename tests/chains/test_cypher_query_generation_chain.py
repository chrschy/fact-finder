from typing import Any, Dict, List
from unittest.mock import ANY, MagicMock

import pytest
from langchain.chains import LLMChain
from langchain.chains.graph_qa.cypher import construct_schema
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompts.prompt import PromptTemplate

from fact_finder.chains.cypher_query_generation_chain import CypherQueryGenerationChain
from fact_finder.config.primekg_predicate_descriptions import PREDICATE_DESCRIPTIONS
from fact_finder.prompt_templates import CYPHER_GENERATION_PROMPT
from tests.chains.helpers import build_llm_mock


def test_cypher_generation_is_called_with_expected_arguments(
    question: str, chain: CypherQueryGenerationChain, cypher_llm: MagicMock, cypher_prompt: str
):
    chain.invoke(input={chain.input_key: question})
    cypher_llm.generate_prompt.assert_called_once_with([cypher_prompt], None, callbacks=ANY)


@pytest.mark.parametrize("how_many", (0, 3))
def test_construct_predicate_descriptions_produces_strings(how_many: int, chain: CypherQueryGenerationChain):
    predicate_descriptions = chain._construct_predicate_descriptions_text(PREDICATE_DESCRIPTIONS[:how_many])
    assert isinstance(predicate_descriptions, str)


def test_construct_predicate_descriptions_produces_expected_number_of_lines(chain: CypherQueryGenerationChain):
    how_many = 3
    predicate_descriptions = chain._construct_predicate_descriptions_text(PREDICATE_DESCRIPTIONS[:how_many])
    lines = predicate_descriptions.split("\n")
    assert len(lines) == how_many + 1


def test_construct_predicate_descriptions_produces_header(chain: CypherQueryGenerationChain):
    how_many = 3
    predicate_descriptions = chain._construct_predicate_descriptions_text(PREDICATE_DESCRIPTIONS[:how_many])
    lines = predicate_descriptions.split("\n")
    assert lines[0] == "Here are some descriptions to the most common relationships:"


def test_construct_predicate_descriptions_produces_empty_string_for_empty_list(chain: CypherQueryGenerationChain):
    predicate_descriptions = chain._construct_predicate_descriptions_text([])
    assert predicate_descriptions == ""


def test_produces_result_with_output_key(chain_with_mocked_llm: CypherQueryGenerationChain, question2: str):
    result = chain_with_mocked_llm(question2)
    assert chain_with_mocked_llm.output_key in result.keys()


def test_question_in_intermediate_steps(chain_with_mocked_llm: CypherQueryGenerationChain, question2: str):
    result = chain_with_mocked_llm(question2)
    assert result["intermediate_steps"][0]["question"] == question2


def test_unchanged_in_result(chain_with_mocked_llm: CypherQueryGenerationChain, question2: str):
    result = chain_with_mocked_llm(question2)
    assert result[chain_with_mocked_llm.input_key] == question2


def test_produces_expected_result(chain_with_mocked_llm: CypherQueryGenerationChain, question2: str):
    result = chain_with_mocked_llm(question2)
    assert (
        result[chain_with_mocked_llm.output_key]
        == "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name"
    )


@pytest.fixture
def chain_with_mocked_llm(chain: CypherQueryGenerationChain, llm_chain: LLMChain) -> CypherQueryGenerationChain:
    chain.cypher_generation_chain = llm_chain
    return chain


@pytest.fixture
def llm_chain(cypher_llm, llm_chain_result) -> LLMChain:
    class LLMChainMock(LLMChain):
        def generate(
            self, input_list: List[Dict[str, Any]], run_manager: CallbackManagerForChainRun | None = None
        ) -> LLMResult:
            return llm_chain_result

    return LLMChainMock(llm=cypher_llm, prompt=CYPHER_GENERATION_PROMPT)


@pytest.fixture
def llm_chain_result() -> LLMResult:
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    text="MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name",
                    generation_info={"finish_reason": "stop", "logprobs": None},
                    message=AIMessage(
                        content="MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name"
                    ),
                )
            ]
        ],
        llm_output={
            "model_name": "gpt-4",
            "token_usage": {"completion_tokens": 27, "prompt_tokens": 1162, "total_tokens": 1189},
        },
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
def question() -> str:
    return "<this is the user question>"


@pytest.fixture
def chain(cypher_llm, structured_schema, cypher_prompt_template) -> CypherQueryGenerationChain:
    chain = CypherQueryGenerationChain(
        llm=cypher_llm, graph_structured_schema=structured_schema, prompt_template=cypher_prompt_template
    )
    return chain


@pytest.fixture
def cypher_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["schema", "question"], template="Generate some cypher with schema:\n{schema}\nFor {question}:"
    )


@pytest.fixture
def cypher_llm(cypher_query) -> BaseLanguageModel:
    return build_llm_mock(cypher_query)


@pytest.fixture(params=["<this is a cypher query>"])
def cypher_query(request) -> str:
    return request.param


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
def question2() -> str:
    return "Which drugs are associated with epilepsy?"
