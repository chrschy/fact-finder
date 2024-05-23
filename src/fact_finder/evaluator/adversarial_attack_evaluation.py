from functools import partial
from typing import Dict, Any

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate

from fact_finder.chains import GraphQAChain, CypherQueryGenerationChain
from fact_finder.chains.filtered_primekg_question_preprocessing_chain import FilteredPrimeKGQuestionPreprocessingChain
from fact_finder.chains.graph_qa_chain import GraphQAChainConfig
from fact_finder.config.primekg_config import (
    _parse_primekg_args,
    _build_preprocessors,
    _get_graph_prompt_templates,
    _get_primekg_entity_categories,
)
from fact_finder.config.primekg_predicate_descriptions import PREDICATE_DESCRIPTIONS
from fact_finder.prompt_templates import KEYWORD_PROMPT, COMBINED_QA_PROMPT
from fact_finder.tools.entity_detector import EntityDetector
from fact_finder.utils import load_chat_model, build_neo4j_graph


QUESTION_WRONG_CYPHER_MAPPING = {}


class AdversarialCypherQueryGenerationChain(CypherQueryGenerationChain):
    """
    Class only used for evaluation of adversarial attacks. It simply returns a wrong cypher query given a natural
    language question. Should downstream test the verbalization of it.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        prompt_template: BasePromptTemplate,
        graph_structured_schema: Dict[str, Any],
        adversarial_question_query_mapping: Dict[str, str],
    ):
        super().__init__(llm, prompt_template, graph_structured_schema)
        self.__adversarial_question_query_mapping = adversarial_question_query_mapping

        ...

    def _generate_cypher(self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun) -> str:
        return self.__adversarial_question_query_mapping[inputs[self.input_key]]


class AdversarialAttackGraphQAChain(GraphQAChain):

    def __init__(self, config: GraphQAChainConfig, adversarial_cypher_chain: AdversarialCypherQueryGenerationChain):
        super().__init__(config)
        self.__adversarial_cypher_chain = adversarial_cypher_chain

    def _build_cypher_query_generation_chain(self, config: GraphQAChainConfig) -> CypherQueryGenerationChain:
        return self.__adversarial_cypher_chain


class AdversarialAttackEvaluation:

    def __init__(self, chain: AdversarialAttackGraphQAChain):
        self.__chain = chain

    def evaluate(self): ...


def build_chain(model: BaseLanguageModel, combine_output_with_sematic_scholar: bool, args: List[str] = []) -> Chain:
    parsed_args = _parse_primekg_args(args)
    graph = build_neo4j_graph()
    cypher_preprocessors = _build_preprocessors(graph, parsed_args.normalized_graph)
    cypher_prompt, answer_generation_prompt = _get_graph_prompt_templates()
    config = GraphQAChainConfig(
        llm=model,
        graph=graph,
        cypher_prompt=cypher_prompt,
        answer_generation_prompt=answer_generation_prompt,
        cypher_query_preprocessors=cypher_preprocessors,
        predicate_descriptions=PREDICATE_DESCRIPTIONS[:10],
        return_intermediate_steps=True,
        use_entity_detection_preprocessing=parsed_args.use_entity_detection_preprocessing,
        entity_detection_preprocessor_type=partial(FilteredPrimeKGQuestionPreprocessingChain, graph=graph),
        entity_detector=EntityDetector() if parsed_args.use_entity_detection_preprocessing else None,
        allowed_types_and_description_templates=_get_primekg_entity_categories(),
        use_subgraph_expansion=parsed_args.use_subgraph_expansion,
        combine_output_with_sematic_scholar=combine_output_with_sematic_scholar,
        semantic_scholar_keyword_prompt=KEYWORD_PROMPT,
        combined_answer_generation_prompt=COMBINED_QA_PROMPT,
    )
    return AdversarialAttackGraphQAChain(
        config,
        AdversarialCypherQueryGenerationChain(
            llm=load_chat_model(),
            prompt_template=config.cypher_prompt,
            graph_structured_schema=config.graph.get_structured_schema,
            adversarial_question_query_mapping=QUESTION_WRONG_CYPHER_MAPPING,
        ),
    )


if __name__ == "__main__":
    ...
