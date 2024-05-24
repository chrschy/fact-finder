from functools import partial
from typing import Dict, Any, List, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from pydantic.v1 import Extra

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
from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluation_samples import manual_samples
from fact_finder.prompt_templates import KEYWORD_PROMPT, COMBINED_QA_PROMPT
from fact_finder.tools.entity_detector import EntityDetector
from fact_finder.utils import load_chat_model, build_neo4j_graph


QUESTION_WRONG_CYPHER_MAPPING = {
    "Which medications have more off-label uses than approved indications?": 'MATCH (d:drug  {name:"lamotrigine"})-[:side_effect]-(e) RETURN e'
}


class AdversarialCypherQueryGenerationChain(Chain):
    """
    Class only used for evaluation of adversarial attacks. It simply returns a wrong cypher query given a natural
    language question. Should downstream test the verbalization of it.
    """
    input_key: str = "question"  #: :meta private:
    output_key: str = "cypher_query"  #: :meta private:
    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["cypher_query"]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        query = self._generate_cypher(inputs)
        return self._prepare_chain_result(inputs=inputs, generated_cypher=query)

    def _prepare_chain_result(self, inputs: Dict[str, Any], generated_cypher: str) -> Dict[str, Any]:
        chain_result = {self.output_key: generated_cypher}

        intermediate_steps = inputs.get("intermediate_steps", [])

        intermediate_steps += [
            {"question": inputs[self.input_key]},
            {self.output_key: generated_cypher},
            {f"{self.__class__.__name__}_filled_prompt": "n/a"},
        ]
        chain_result["intermediate_steps"] = intermediate_steps
        return chain_result

    def _generate_cypher(self, inputs: Dict[str, Any]) -> str:
        return QUESTION_WRONG_CYPHER_MAPPING[inputs["question"]]


class AdversarialAttackGraphQAChain(GraphQAChain):

    def _build_cypher_query_generation_chain(self, config: GraphQAChainConfig) -> AdversarialCypherQueryGenerationChain:
        return AdversarialCypherQueryGenerationChain()


class AdversarialAttackEvaluation:

    def __init__(self, chain: AdversarialAttackGraphQAChain):
        self.__chain = chain

    def evaluate(self): ...


def build_chain(args: List[str] = []) -> Chain:
    parsed_args = _parse_primekg_args(args)
    graph = build_neo4j_graph()
    cypher_preprocessors = _build_preprocessors(graph, parsed_args.normalized_graph)
    cypher_prompt, answer_generation_prompt = _get_graph_prompt_templates()
    config = GraphQAChainConfig(
        llm=load_chat_model(),
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
        combine_output_with_sematic_scholar=False,
        semantic_scholar_keyword_prompt=KEYWORD_PROMPT,
        combined_answer_generation_prompt=COMBINED_QA_PROMPT,
    )

    return AdversarialAttackGraphQAChain(config=config)


def eval_samples(limit_of_samples: int = None):
    eval_samples = []
    for manual_sample in manual_samples[:limit_of_samples]:
        eval_sample = EvaluationSample(
            question=manual_sample["question"],
            cypher_query=manual_sample["expected_cypher"],
            expected_answer=manual_sample["expected_answer"],
            nodes=manual_sample["nodes"],
        )
        eval_samples.append(eval_sample)
    return eval_samples


if __name__ == "__main__":
    chain = build_chain(args=["--normalized_graph", "--use_entity_detection_preprocessing"])
    # chain = build_chain(args=["--normalized_graph"])
    samples = eval_samples(limit_of_samples=1)
    for sample in samples:
        print(sample.question)
        inputs = {"question": sample.question}
        result = chain.invoke(inputs)
        ...
