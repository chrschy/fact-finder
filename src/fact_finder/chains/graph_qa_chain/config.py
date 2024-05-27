from typing import Any, Dict, List, Optional

from fact_finder.chains.entity_detection_question_preprocessing_chain import (
    EntityDetectionQuestionPreprocessingProtocol,
)
from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import (
    CypherQueryPreprocessor,
)
from fact_finder.tools.entity_detector import EntityDetector
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from pydantic.v1 import BaseModel, root_validator


class GraphQAChainConfig(BaseModel):

    class Config:
        arbitrary_types_allowed: bool = True

    llm: BaseLanguageModel
    graph: Neo4jGraph
    cypher_prompt: BasePromptTemplate
    answer_generation_prompt: BasePromptTemplate
    cypher_query_preprocessors: List[CypherQueryPreprocessor]
    predicate_descriptions: List[Dict[str, str]] = []
    schema_error_string: str = "SCHEMA_ERROR"
    return_intermediate_steps: bool = True

    use_entity_detection_preprocessing: bool = False
    entity_detection_preprocessor_type: Optional[
        EntityDetectionQuestionPreprocessingProtocol
    ]  # = EntityDetectionQuestionPreprocessingChain
    entity_detector: Optional[EntityDetector] = None
    # The keys of this dict contain the (lower case) type names for which entities can be replaced.
    # They map to a template explaining the type of an entity (marked via {entity})
    # Example: "chemical_compounds", "{entity} is a chemical compound."
    allowed_types_and_description_templates: Dict[str, str] = {}

    skip_subgraph_generation: bool = False
    use_subgraph_expansion: bool = False

    combine_output_with_sematic_scholar: bool = False
    semantic_scholar_keyword_prompt: Optional[BasePromptTemplate] = None
    combined_answer_generation_prompt: Optional[BasePromptTemplate] = None
    top_k: int = 2000  # todo change back when not evaluating

    @root_validator(allow_reuse=True)
    def check_entity_detection_preprocessing_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["use_entity_detection_preprocessing"] and values.get("entity_detector") is None:
            raise ValueError("When setting use_entity_detection_preprocessing, an entity_detector has to be provided.")
        if values["use_entity_detection_preprocessing"] and values.get("entity_detection_preprocessor_type") is None:
            raise ValueError(
                "When setting use_entity_detection_preprocessing, "
                "an entity_detection_preprocessor_type has to be provided."
            )
        if (
            values["use_entity_detection_preprocessing"]
            and len(values.get("allowed_types_and_description_templates", {})) == 0
        ):
            raise ValueError(
                "When setting use_entity_detection_preprocessing, "
                "allowed_types_and_description_templates has to be provided."
            )
        return values

    @root_validator(allow_reuse=True)
    def check_semantic_scholar_configured_correctly(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["combine_output_with_sematic_scholar"] and values.get("semantic_scholar_keyword_prompt") is None:
            raise ValueError(
                "When setting combine_output_with_sematic_scholar, "
                "a semantic_scholar_keyword_prompt has to be provided."
            )
        if values["combine_output_with_sematic_scholar"] and values.get("combined_answer_generation_prompt") is None:
            raise ValueError(
                "When setting combine_output_with_sematic_scholar, "
                "combined_answer_generation_prompt has to be provided."
            )
        return values
