from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from langchain.chains import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompts import PromptTemplate

from fact_finder.chains.answer_generation_chain import AnswerGenerationChain
from fact_finder.prompt_templates import CYPHER_QA_PROMPT


def test_qa_chain(
    answer_generation_chain: AnswerGenerationChain, output_from_graph_chain: Dict[str, Any], expected_answer
):
    result = answer_generation_chain.invoke(input=output_from_graph_chain)
    assert answer_generation_chain.output_key in result.keys()
    assert result[answer_generation_chain.output_key] == expected_answer


@pytest.fixture
def answer_generation_chain(llm: BaseChatModel, llm_chain: LLMChain) -> AnswerGenerationChain:
    answer_generation_chain = AnswerGenerationChain(llm=llm, prompt_template=CYPHER_QA_PROMPT)
    answer_generation_chain.llm_chain = llm_chain
    return answer_generation_chain


@pytest.fixture
def llm() -> BaseChatModel:
    return MagicMock(spec=BaseChatModel)


@pytest.fixture
def prompt() -> PromptTemplate:
    return MagicMock(spec=PromptTemplate)


@pytest.fixture
def llm_chain(llm: BaseChatModel, prompt: PromptTemplate) -> LLMChain:
    return MockedLLMChain(llm=llm, prompt=prompt)


@pytest.fixture
def output_from_graph_chain() -> Dict[str, Any]:
    return {
        "question": "Which drugs are associated with epilepsy?",
        "cypher_query": "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name",
        "intermediate_steps": [
            {"question": "Which drugs are associated with epilepsy?"},
            {
                "FormatPreprocessor": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
                "LowerCasePropertiesCypherQueryPreprocessor": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
            },
            {
                "graph_result": [
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
            },
        ],
        "preprocessed_cypher_query": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
        "graph_result": [
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
        ],
    }


@pytest.fixture
def expected_answer():
    return "The drugs associated with epilepsy are phenytoin, valproic acid, lamotrigine, diazepam, clonazepam, fosphenytoin, mephenytoin, neocitrullamon, carbamazepine, phenobarbital, secobarbital, primidone, and lorazepam."


class MockedLLMChain(LLMChain):
    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        return LLMResult(
            generations=[
                [
                    ChatGeneration(
                        text="The drugs associated with epilepsy are phenytoin, valproic acid, lamotrigine, diazepam, clonazepam, fosphenytoin, mephenytoin, neocitrullamon, carbamazepine, phenobarbital, secobarbital, primidone, and lorazepam.",
                        generation_info={"finish_reason": "stop", "logprobs": None},
                        message=AIMessage(
                            content="The drugs associated with epilepsy are phenytoin, valproic acid, lamotrigine, diazepam, clonazepam, fosphenytoin, mephenytoin, neocitrullamon, carbamazepine, phenobarbital, secobarbital, primidone, and lorazepam."
                        ),
                    )
                ]
            ],
        )
