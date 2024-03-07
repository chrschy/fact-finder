from typing import Any, Dict, List

import pytest
from fact_finder.chains.cypher_query_preprocessors_chain import (
    CypherQueryPreprocessorsChain,
)
from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import (
    CypherQueryPreprocessor,
)
from fact_finder.tools.cypher_preprocessors.format_preprocessor import (
    FormatPreprocessor,
)
from fact_finder.tools.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)


def test_produces_expected_output_key(
    cypher_query_generation_chain_result: Dict[str, Any], preprocessors: List[CypherQueryPreprocessor]
):
    chain = CypherQueryPreprocessorsChain(cypher_query_preprocessors=preprocessors)
    result = chain(cypher_query_generation_chain_result)
    assert chain.output_key in result.keys()


def test_produces_expected_number_of_intermediate_steps(
    cypher_query_generation_chain_result: Dict[str, Any], preprocessors: List[CypherQueryPreprocessor]
):
    chain = CypherQueryPreprocessorsChain(cypher_query_preprocessors=preprocessors, return_intermediate_steps=True)
    result = chain(cypher_query_generation_chain_result)
    assert len(result["intermediate_steps"]) == len(preprocessors) + len(
        cypher_query_generation_chain_result["intermediate_steps"]
    )


def test_applies_expected_preprocessings(
    cypher_query_generation_chain_result: Dict[str, Any], preprocessors: List[CypherQueryPreprocessor]
):
    chain = CypherQueryPreprocessorsChain(cypher_query_preprocessors=preprocessors)
    result = chain(cypher_query_generation_chain_result)
    assert (
        result["preprocessed_cypher_query"]
        == 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name'
    )


def test_preprocessors_are_called_in_order(cypher_query_generation_chain_result: Dict[str, Any]):

    class CypherQueryPreprocessorMock(CypherQueryPreprocessor):
        call_idx = 0

        def __init__(self, expected_idx: int) -> None:
            self._expected_idx = expected_idx

        def __call__(self, cypher_query: str) -> str:
            assert CypherQueryPreprocessorMock.call_idx == self._expected_idx
            CypherQueryPreprocessorMock.call_idx += 1
            return cypher_query

    num_preprocs = 3
    preprocs = [CypherQueryPreprocessorMock(expected_idx=i) for i in range(num_preprocs)]
    chain = CypherQueryPreprocessorsChain(cypher_query_preprocessors=preprocs)
    chain(cypher_query_generation_chain_result)
    assert CypherQueryPreprocessorMock.call_idx == num_preprocs


@pytest.fixture
def cypher_query_generation_chain_result() -> Dict[str, Any]:
    return {
        "cypher_query": "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name",
        "intermediate_steps": [{"question": "Which drugs are associated with epilepsy?"}],
    }


@pytest.fixture
def preprocessors() -> List[CypherQueryPreprocessor]:
    return [FormatPreprocessor(), LowerCasePropertiesCypherQueryPreprocessor()]
