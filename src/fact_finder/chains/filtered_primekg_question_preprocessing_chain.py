from typing import Dict, List, Tuple

from fact_finder.chains.entity_detection_question_preprocessing_chain import (
    EntityDetectionQuestionPreprocessingChain,
)
from fact_finder.tools.entity_detector import EntityDetector
from langchain_community.graphs import Neo4jGraph


class FilteredPrimeKGQuestionPreprocessingChain(EntityDetectionQuestionPreprocessingChain):
    graph: Neo4jGraph
    _side_effect_exists_cypher_query: str = (
        'MATCH(node:effect_or_phenotype {name: "{entity_name}"}) RETURN COUNT(node) > 0 AS exists'
    )

    def __init__(
        self,
        *,
        entity_detector: EntityDetector,
        allowed_types_and_description_templates: Dict[str, str],
        graph: Neo4jGraph,
        return_intermediate_steps: bool = True,
    ):
        allowed_types_and_description_templates["side_effect"] = "{entity} is a disease or a effect_or_phenotype."
        super().__init__(
            entity_detector=entity_detector,
            allowed_types_and_description_templates=allowed_types_and_description_templates,
            return_intermediate_steps=return_intermediate_steps,
            graph=graph,
        )

    def _extract_entity_data(
        self, question: str, entity_results: List[Tuple[int, int, str, str]]
    ) -> Tuple[str, List[str]]:
        entity_type_hints = []
        new_question = ""
        last_index = 0
        for start, end, pref_name, type in entity_results:
            type = type.lower()
            if type == "disease":
                pref_name, type = self._filter_diseases_that_are_also_side_effects(question[start:end], pref_name, type)
            new_question += question[last_index:start] + pref_name
            last_index = end
            entity_type_hints.append(self._create_type_hint(pref_name, type))
        new_question += question[last_index:]
        return new_question, entity_type_hints

    def _filter_diseases_that_are_also_side_effects(
        self, original_name: str, pref_name: str, type: str
    ) -> Tuple[str, str]:
        if self._exists_as_side_effect_node(original_name):
            type = "side_effect"
            pref_name = original_name
        elif self._exists_as_side_effect_node(pref_name):
            type = "side_effect"
        return pref_name, type

    def _exists_as_side_effect_node(self, name: str) -> bool:
        cypher_query = self._side_effect_exists_cypher_query.replace("{entity_name}", name)
        nodes = self.graph.query(cypher_query)
        return nodes[0]["exists"]
