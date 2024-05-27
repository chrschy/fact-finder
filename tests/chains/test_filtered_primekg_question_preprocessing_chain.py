from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import pytest
from fact_finder.chains.filtered_primekg_question_preprocessing_chain import (
    FilteredPrimeKGQuestionPreprocessingChain,
)
from fact_finder.tools.entity_detector import EntityDetector
from langchain_community.graphs import Neo4jGraph
from tests.chains.test_entity_detection_question_preprocessing_chain import (
    _ILLEGAL_CATEGORIES_ENTITIES,
    _NON_OVERLAPPING_ENTITIES,
    _PARTIAL_OVERLAPPING_ENTITIES,
    _SUBSET_OVERLAP_ENTITIES,
)

_DISEASE_ENTITY: List[Dict[str, int | str]] = [
    {"start_span": 0, "end_span": 2, "pref_term": "pref_name1", "sem_type": "category1"},
    {"start_span": 4, "end_span": 6, "pref_term": "disease_pref_name", "sem_type": "Disease"},
    {"start_span": 6, "end_span": 8, "pref_term": "pref_name3", "sem_type": "category1"},
]


def test_produces_expected_output_key(inputs: Dict[str, str], chain: FilteredPrimeKGQuestionPreprocessingChain):
    res = chain.invoke(inputs)
    assert all(k in res for k in chain.output_keys)


def test_result_contains_entity_replacements(chain_result: str, entities: List[Dict[str, int | str]]):
    assert all(e["pref_term"] in chain_result for e in entities)


def test_result_contains_entity_hints(
    chain_result: str, entities: List[Dict[str, int | str]], allowed_categories: Dict[str, str]
):
    hints = [allowed_categories[e["sem_type"]].replace("{entity}", e["pref_term"]).capitalize() for e in entities]
    assert all(h in chain_result for h in hints)


@pytest.mark.parametrize("entities", (_ILLEGAL_CATEGORIES_ENTITIES,), indirect=True)
def test_result_only_contains_entities_from_allowed_categories(
    chain_result: str, entities: List[Dict[str, int | str]], allowed_categories: Dict[str, str]
):
    assert all((e["pref_term"] in chain_result) == (e["sem_type"] in allowed_categories) for e in entities)


@pytest.mark.parametrize("entities", (_PARTIAL_OVERLAPPING_ENTITIES,), indirect=True)
def test_result_does_not_contain_replacements_for_overlapping_entities(chain_result: str):
    assert "pref_name2" not in chain_result and "pref_name3" not in chain_result


@pytest.mark.parametrize("entities", _SUBSET_OVERLAP_ENTITIES, indirect=True)
def test_result_does_not_contain_entities_that_are_contained_in_a_larger_entity(chain_result: str):
    assert "pref_name3" not in chain_result


@pytest.mark.parametrize("entities", _SUBSET_OVERLAP_ENTITIES, indirect=True)
def test_result_contains_entity_that_contains_smaller_entity(chain_result: str):
    assert "pref_name2" in chain_result


@pytest.mark.parametrize("entities", (_DISEASE_ENTITY,), indirect=True)
def test_disease_replaced_normally_if_no_matching_side_effect_found(chain_result: str):
    assert "disease_pref_name" in chain_result


@pytest.mark.parametrize("entities", (_DISEASE_ENTITY,), indirect=True)
def test_normal_disease_entity_hint_if_no_matching_side_effect_found(chain_result: str):
    assert "Disease_pref_name is a disease." in chain_result


@pytest.mark.parametrize("entities,graph_result", [(_DISEASE_ENTITY, True)], indirect=True)
def test_disease_not_replaced_if_matching_side_effect_found_for_original_name(chain_result: str):
    assert "disease_pref_name" not in chain_result and "e2" in chain_result


@pytest.mark.parametrize("entities,graph_result", [(_DISEASE_ENTITY, True)], indirect=True)
def test_correct_entity_hint_if_matching_side_effect_found_for_original_name(chain_result: str):
    assert "E2 is a disease or a effect_or_phenotype." in chain_result


@pytest.mark.parametrize("entities,graph_result", [(_DISEASE_ENTITY, (False, False, True, False))], indirect=True)
def test_disease_replaced_if_matching_side_effect_found_for_preferred_name(chain_result: str):
    assert "disease_pref_name" in chain_result and "e2" not in chain_result


@pytest.mark.parametrize("entities,graph_result", [(_DISEASE_ENTITY, (False, False, True, False))], indirect=True)
def test_correct_entity_hint_if_matching_side_effect_found_for_preferred_name(chain_result: str):
    assert "Disease_pref_name is a disease or a effect_or_phenotype." in chain_result


@pytest.fixture
def chain_result(inputs: Dict[str, str], chain: FilteredPrimeKGQuestionPreprocessingChain) -> str:
    return chain.invoke(inputs)[chain.output_keys[0]]


@pytest.fixture
def inputs(chain: FilteredPrimeKGQuestionPreprocessingChain) -> Dict[str, str]:
    return {chain.input_keys[0]: "e1  e2e3 bla?"}


@pytest.fixture
def chain(
    entity_detector: EntityDetector, allowed_categories: Dict[str, str], graph
) -> FilteredPrimeKGQuestionPreprocessingChain:
    return FilteredPrimeKGQuestionPreprocessingChain(
        entity_detector=entity_detector, allowed_types_and_description_templates=allowed_categories, graph=graph
    )


@pytest.fixture
def allowed_categories() -> Dict[str, str]:
    return {
        "category1": "{entity} is in category1.",
        "category2": "{entity} is in category2.",
        "category3": "{entity} is in category3.",
        "disease": "{entity} is a disease.",
    }


@pytest.fixture
def graph(graph_result: List[Dict[str, bool]] | List[List[Dict[str, bool]]]) -> Neo4jGraph:
    graph = MagicMock(spec=Neo4jGraph)
    graph.query = MagicMock()
    if isinstance(graph_result[0], dict):
        graph.query.return_value = graph_result
    else:
        graph.query.side_effect = graph_result
    return graph


@pytest.fixture
def graph_result(request: pytest.FixtureRequest) -> List[Dict[str, bool]] | List[List[Dict[str, bool]]]:
    if not hasattr(request, "param") or request.param is None:
        return [{"exists": False}]
    if isinstance(request.param, bool):
        return [{"exists": request.param}]
    return [[{"exists": p}] for p in request.param]


@pytest.fixture
def entity_detector(entities: Dict[str, int | str]) -> EntityDetector:
    det = MagicMock(spec=EntityDetector)
    det.return_value = entities
    return det


@pytest.fixture
def entities(request: pytest.FixtureRequest) -> List[Dict[str, int | str]]:
    if not hasattr(request, "param") or request.param is None:
        return _NON_OVERLAPPING_ENTITIES
    return request.param
