from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import pytest
from fact_finder.chains.entity_detection_question_preprocessing_chain import (
    EntityDetectionQuestionPreprocessingChain,
)
from fact_finder.tools.entity_detector import EntityDetector

_NON_OVERLAPPING_ENTITIES: List[Dict[str, int | str]] = [
    {"start_span": 0, "end_span": 2, "pref_term": "pref_name1", "sem_type": "category1"},
    {"start_span": 4, "end_span": 6, "pref_term": "pref_name2", "sem_type": "category2"},
    {"start_span": 6, "end_span": 8, "pref_term": "pref_name3", "sem_type": "category1"},
]

_ILLEGAL_CATEGORIES_ENTITIES: List[Dict[str, int | str]] = [
    {"start_span": 0, "end_span": 2, "pref_term": "pref_name1", "sem_type": "category1"},
    {"start_span": 4, "end_span": 6, "pref_term": "pref_name2", "sem_type": "category4"},
    {"start_span": 6, "end_span": 8, "pref_term": "pref_name3", "sem_type": "category5"},
]

_PARTIAL_OVERLAPPING_ENTITIES: List[Dict[str, int | str]] = [
    {"start_span": 0, "end_span": 2, "pref_term": "pref_name1", "sem_type": "category1"},
    {"start_span": 4, "end_span": 6, "pref_term": "pref_name2", "sem_type": "category1"},
    {"start_span": 5, "end_span": 8, "pref_term": "pref_name3", "sem_type": "category1"},
]

_SUBSET_OVERLAP_ENTITIES: Tuple[List[Dict[str, int | str]], List[Dict[str, int | str]]] = (
    [
        {"start_span": 0, "end_span": 2, "pref_term": "pref_name1", "sem_type": "category1"},
        {"start_span": 4, "end_span": 8, "pref_term": "pref_name2", "sem_type": "category1"},
        {"start_span": 5, "end_span": 8, "pref_term": "pref_name3", "sem_type": "category1"},
    ],
    [
        {"start_span": 0, "end_span": 2, "pref_term": "pref_name1", "sem_type": "category1"},
        {"start_span": 4, "end_span": 7, "pref_term": "pref_name3", "sem_type": "category1"},
        {"start_span": 4, "end_span": 8, "pref_term": "pref_name2", "sem_type": "category1"},
    ],
)


def test_produces_expected_output_key(inputs: Dict[str, str], chain: EntityDetectionQuestionPreprocessingChain):
    res = chain.invoke(inputs)
    assert all(k in res for k in chain.output_keys)


def test_result_contains_entity_replacements(
    inputs: Dict[str, str], chain: EntityDetectionQuestionPreprocessingChain, entities: List[Dict[str, int | str]]
):
    res = chain.invoke(inputs)
    assert all(e["pref_term"] in res[chain.output_keys[0]] for e in entities)


def test_result_contains_entity_hints(
    inputs: Dict[str, str],
    chain: EntityDetectionQuestionPreprocessingChain,
    entities: List[Dict[str, int | str]],
    allowed_categories: Dict[str, str],
):
    res = chain.invoke(inputs)
    hints = [allowed_categories[e["sem_type"]].replace("{entity}", e["pref_term"]).capitalize() for e in entities]
    assert all(h in res[chain.output_keys[0]] for h in hints)


@pytest.mark.parametrize("entities", (_ILLEGAL_CATEGORIES_ENTITIES,), indirect=True)
def test_result_only_contains_entities_from_allowed_categories(
    inputs: Dict[str, str],
    chain: EntityDetectionQuestionPreprocessingChain,
    entities: List[Dict[str, int | str]],
    allowed_categories: Dict[str, str],
):
    res = chain.invoke(inputs)
    assert all((e["pref_term"] in res[chain.output_keys[0]]) == (e["sem_type"] in allowed_categories) for e in entities)


@pytest.mark.parametrize("entities", (_PARTIAL_OVERLAPPING_ENTITIES,), indirect=True)
def test_result_does_not_contain_replacements_for_overlapping_entities(
    inputs: Dict[str, str], chain: EntityDetectionQuestionPreprocessingChain
):
    res = chain.invoke(inputs)
    assert "pref_name2" not in res[chain.output_keys[0]] and "pref_name3" not in res[chain.output_keys[0]]


@pytest.mark.parametrize("entities", _SUBSET_OVERLAP_ENTITIES, indirect=True)
def test_result_does_not_contain_entities_that_are_contained_in_a_larger_entity(
    inputs: Dict[str, str], chain: EntityDetectionQuestionPreprocessingChain
):
    res = chain.invoke(inputs)
    assert "pref_name3" not in res[chain.output_keys[0]]


@pytest.mark.parametrize("entities", _SUBSET_OVERLAP_ENTITIES, indirect=True)
def test_result_contains_entity_that_contains_smaller_entity(
    inputs: Dict[str, str], chain: EntityDetectionQuestionPreprocessingChain
):
    res = chain.invoke(inputs)
    assert "pref_name2" in res[chain.output_keys[0]]


@pytest.fixture
def inputs(chain: EntityDetectionQuestionPreprocessingChain) -> Dict[str, str]:
    return {chain.input_keys[0]: "e1  e2e3 bla?"}


@pytest.fixture
def chain(
    entity_detector: EntityDetector, allowed_categories: Dict[str, str]
) -> EntityDetectionQuestionPreprocessingChain:
    return EntityDetectionQuestionPreprocessingChain(
        entity_detector=entity_detector, allowed_types_and_description_templates=allowed_categories
    )


@pytest.fixture
def allowed_categories() -> Dict[str, str]:
    return {
        "category1": "{entity} is in category1.",
        "category2": "{entity} is in category2.",
        "category3": "{entity} is in category3.",
    }


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
