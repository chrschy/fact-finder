from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock

import pytest
from langchain.chains import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain_core.prompts import PromptTemplate

from fact_finder.chains.answer_generation_chain import AnswerGenerationChain


def test_qa_chain(qa_chain, output_from_graph_chain, expected_answer):
    result = qa_chain(inputs=output_from_graph_chain)
    assert qa_chain.output_key in result.keys()
    assert result[qa_chain.output_key] == expected_answer


@pytest.fixture
def qa_chain(llm, custom_llm_chain):
    qa_chain = AnswerGenerationChain(llm=llm)
    qa_chain.llm_chain = custom_llm_chain
    return qa_chain


@pytest.fixture
def llm():
    return MagicMock(spec=BaseChatModel)


@pytest.fixture
def prompt():
    return MagicMock(spec=PromptTemplate)


@pytest.fixture
def custom_llm_chain(llm, prompt):
    return MockedLLMChain(llm=llm, prompt=prompt)


@pytest.fixture
def output_from_graph_chain():
    return {
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
