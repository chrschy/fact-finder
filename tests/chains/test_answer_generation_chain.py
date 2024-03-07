from typing import Any, Dict, List
from unittest.mock import ANY, MagicMock

import pytest
from fact_finder.chains.answer_generation_chain import AnswerGenerationChain
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.string import StringPromptValue
from tests.chains.helpers import build_llm_mock


def test_output_key_in_result(
    answer_generation_chain: AnswerGenerationChain,
    output_from_graph_chain: Dict[str, Any],
):
    result = answer_generation_chain.invoke(input=output_from_graph_chain)
    assert answer_generation_chain.output_key in result.keys()


def test_returned_result_matches_model_output(
    answer_generation_chain: AnswerGenerationChain, output_from_graph_chain: Dict[str, Any], llm_answer: str
):
    result = answer_generation_chain.invoke(input=output_from_graph_chain)
    assert result[answer_generation_chain.output_key] == llm_answer


def test_answer_generation_llm_is_called_with_expected_arguments(
    answer_generation_chain: AnswerGenerationChain,
    llm: BaseLanguageModel,
    generation_prompt: str,
    output_from_graph_chain: Dict[str, Any],
):
    answer_generation_chain.invoke(input=output_from_graph_chain)
    llm.generate_prompt.assert_called_once_with([generation_prompt], None, callbacks=ANY)


@pytest.fixture
def generation_prompt(
    prompt_template: PromptTemplate, graph_result: List[Dict[str, str]], question: str
) -> StringPromptValue:
    return prompt_template.format_prompt(context=graph_result, question=question)


@pytest.fixture
def answer_generation_chain(llm: BaseChatModel, prompt_template: PromptTemplate) -> AnswerGenerationChain:
    answer_generation_chain = AnswerGenerationChain(llm=llm, prompt_template=prompt_template)
    return answer_generation_chain


@pytest.fixture
def prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"], template="Generate an answer to {question} using {context}:"
    )


@pytest.fixture
def llm(llm_answer) -> BaseLanguageModel:
    return build_llm_mock(llm_answer)


@pytest.fixture
def llm_answer() -> str:
    return (
        "The drugs associated with epilepsy are phenytoin, valproic acid, "
        "lamotrigine, diazepam, clonazepam, fosphenytoin, mephenytoin, "
        "neocitrullamon, carbamazepine, phenobarbital, secobarbital, "
        "primidone, and lorazepam."
    )


@pytest.fixture
def prompt() -> PromptTemplate:
    return MagicMock(spec=PromptTemplate)


@pytest.fixture
def output_from_graph_chain(question: str, graph_result: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "question": question,
        "cypher_query": "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name",
        "intermediate_steps": [
            {"question": "Which drugs are associated with epilepsy?"},
            {
                "FormatPreprocessor": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
                "LowerCasePropertiesCypherQueryPreprocessor": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
            },
            {"graph_result": graph_result},
        ],
        "preprocessed_cypher_query": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
        "graph_result": graph_result,
    }


@pytest.fixture
def question() -> str:
    return "Which drugs are associated with epilepsy?"


@pytest.fixture
def graph_result() -> List[Dict[str, str]]:
    return [
        {"d.name": "phenytoin"},
        {"d.name": "phenytoin"},
        {"d.name": "phenytoin"},
        {"d.name": "valproic acid"},
        {"d.name": "lamotrigine"},
        {"d.name": "lamotrigine"},
        {"d.name": "diazepam"},
        {"d.name": "clonazepam"},
        {"d.name": "fosphenytoin"},
        {"d.name": "mephenytoin"},
        {"d.name": "mephenytoin"},
        {"d.name": "neocitrullamon"},
        {"d.name": "carbamazepine"},
        {"d.name": "carbamazepine"},
        {"d.name": "phenobarbital"},
        {"d.name": "phenobarbital"},
        {"d.name": "secobarbital"},
        {"d.name": "primidone"},
        {"d.name": "primidone"},
        {"d.name": "lorazepam"},
    ]
