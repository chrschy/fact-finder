from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration

from fact_finder.chains.custom_llm_chain import CustomLLMChain
from fact_finder.chains.qa_chain import QAChain
from fact_finder.prompt_templates import CYPHER_QA_PROMPT


class MockedCustomLLMChain(CustomLLMChain):
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
            llm_output={
                "token_usage": {"completion_tokens": 74, "prompt_tokens": 335, "total_tokens": 409},
                "model_name": "gpt-4",
            },
        )


def test_qa_chain(output_from_graph_chain, expected_answer, qa_chain):
    result = qa_chain(inputs=output_from_graph_chain)
    assert result[qa_chain.output_key] == expected_answer
    assert qa_chain.output_key in result.keys()
    assert len(result[qa_chain.intermediate_steps_key]) == 4


@pytest.fixture
def qa_chain(llm_mocked, return_intermediate_steps):
    return QAChain(
        llm_chain=MockedCustomLLMChain(llm=llm_mocked, prompt=CYPHER_QA_PROMPT),
        return_intermediate_steps=return_intermediate_steps,
    )


@pytest.fixture
def llm_mocked():
    return MagicMock(spec=BaseChatModel)


@pytest.fixture
def return_intermediate_steps():
    return True


@pytest.fixture
def expected_answer():
    return "The drugs associated with epilepsy are phenytoin, valproic acid, lamotrigine, diazepam, clonazepam, fosphenytoin, mephenytoin, neocitrullamon, carbamazepine, phenobarbital, secobarbital, primidone, and lorazepam."


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
