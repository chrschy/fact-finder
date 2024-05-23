from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fact_finder.chains.answer_generation_chain import AnswerGenerationChain
from fact_finder.chains.combined_graph_rag_qa_chain import CombinedQAChain
from fact_finder.chains.cypher_query_generation_chain import CypherQueryGenerationChain
from fact_finder.chains.cypher_query_preprocessors_chain import (
    CypherQueryPreprocessorsChain,
)
from fact_finder.chains.entity_detection_question_preprocessing_chain import (
    EntityDetectionQuestionPreprocessingChain,
)
from fact_finder.chains.graph_chain import GraphChain
from fact_finder.chains.graph_qa_chain.config import GraphQAChainConfig
from fact_finder.chains.graph_qa_chain.early_stopping_chain import (
    GraphQAChainEarlyStopping,
)
from fact_finder.chains.graph_qa_chain.output import GraphQAChainOutput
from fact_finder.chains.graph_qa_chain.output_chain import GraphQAChainOutputPreparation
from fact_finder.chains.rag.semantic_scholar_chain import SemanticScholarChain
from fact_finder.chains.subgraph_extractor_chain import SubgraphExtractorChain
from fact_finder.tools.semantic_scholar_search_api_wrapper import (
    SemanticScholarSearchApiWrapper,
)
from fact_finder.tools.subgraph_extension import SubgraphExpansion
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
    RunnableSerializable,
)


class GraphQAChain(Chain):
    schema_error_string: str
    combined_chain: RunnableSequence
    return_intermediate_steps: bool
    input_key: str = "question"  #: :meta private:
    output_key: str = "graph_qa_output"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return [self.output_key]

    def __init__(self, config: GraphQAChainConfig):
        combined_chain = self._build_chain(config)
        super().__init__(
            schema_error_string=config.schema_error_string,
            combined_chain=combined_chain,
            return_intermediate_steps=config.return_intermediate_steps,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, GraphQAChainOutput | List[Dict[str, Any]]]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        config = RunnableConfig(callbacks=_run_manager.get_child())
        chain_result = self.combined_chain.invoke(inputs[self.input_key], config=config)
        result = {self.output_key: chain_result[self.output_key]}
        if self.return_intermediate_steps:
            result[self.intermediate_steps_key] = chain_result[self.intermediate_steps_key]
        return result

    def _build_chain(self, config: GraphQAChainConfig):
        semantic_scholar_chain = self._build_semantic_scholar_chain(config)
        question_input_chain, cypher_gen_key = self._build_question_input_chain(config, semantic_scholar_chain)
        cypher_query_preprocessors_chain = self._build_cypher_query_preprocessors_chain(config)
        graph_chain = self._build_graph_chain(config)
        answer_generation_chain = self._build_answer_generation_chain(config, semantic_scholar_chain)
        subgraph_extractor_chain = self._subgraph_extractor_chain(config)
        output_chain = self._build_output_chain(config)

        parallel_subgraph_and_answer = RunnableParallel(
            **{
                subgraph_extractor_chain.output_key: subgraph_extractor_chain,
                answer_generation_chain.output_key: graph_chain | answer_generation_chain,
            }
        )

        def route(inputs):
            if inputs[cypher_gen_key].startswith(config.schema_error_string):
                return GraphQAChainEarlyStopping(schema_error_string=config.schema_error_string)
            return cypher_query_preprocessors_chain | parallel_subgraph_and_answer | output_chain

        return question_input_chain | RunnableLambda(route)

    def _build_semantic_scholar_chain(self, config: GraphQAChainConfig) -> Optional[SemanticScholarChain]:
        if config.combine_output_with_sematic_scholar:
            assert config.semantic_scholar_keyword_prompt is not None
            return SemanticScholarChain(
                semantic_scholar_search=SemanticScholarSearchApiWrapper(),
                llm=config.llm,
                keyword_prompt_template=config.semantic_scholar_keyword_prompt,
                return_intermediate_steps=config.return_intermediate_steps,
            )
        return None

    def _build_question_input_chain(
        self, config: GraphQAChainConfig, semantic_scholar_chain: Optional[SemanticScholarChain]
    ) -> Tuple[RunnableSerializable, str]:
        cypher_query_generation_chain = self._build_cypher_query_generation_chain(config)
        question_input_chain = self._add_question_preprocessing_chain(config, cypher_query_generation_chain)
        question_input_chain = self._add_parallel_semantic_scholar_chain(
            question_input_chain, semantic_scholar_chain, cypher_query_generation_chain
        )
        return question_input_chain, cypher_query_generation_chain.output_key

    def _build_cypher_query_generation_chain(self, config: GraphQAChainConfig) -> CypherQueryGenerationChain:
        return CypherQueryGenerationChain(
            llm=config.llm,
            graph_structured_schema=config.graph.get_structured_schema,
            prompt_template=config.cypher_prompt,
            predicate_descriptions=config.predicate_descriptions,
            return_intermediate_steps=config.return_intermediate_steps,
        )

    def _add_question_preprocessing_chain(
        self, config: GraphQAChainConfig, question_input_chain: CypherQueryGenerationChain
    ) -> RunnableSerializable:
        if config.use_entity_detection_preprocessing:
            assert config.entity_detector is not None
            preprocessing_chain = EntityDetectionQuestionPreprocessingChain(
                entity_detector=config.entity_detector,
                allowed_types_and_description_templates=config.allowed_types_and_description_templates,
                return_intermediate_steps=config.return_intermediate_steps,
            )
            question_input_chain.input_key = preprocessing_chain.output_key
            return preprocessing_chain | question_input_chain
        return question_input_chain

    def _add_parallel_semantic_scholar_chain(
        self,
        question_input_chain: RunnableSerializable,
        semantic_scholar_chain: Optional[SemanticScholarChain],
        cypher_query_generation_chain: CypherQueryGenerationChain,
    ) -> RunnableSerializable:
        if semantic_scholar_chain is None:
            return question_input_chain

        def _flatten_output(outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
            intermediate_steps = (
                outputs[cypher_query_generation_chain.output_key][cypher_query_generation_chain.intermediate_steps_key]
                + outputs[semantic_scholar_chain.output_key][semantic_scholar_chain.intermediate_steps_key]
            )
            new_output = outputs[semantic_scholar_chain.output_key]
            new_output.update(outputs[cypher_query_generation_chain.output_key])
            new_output[cypher_query_generation_chain.intermediate_steps_key] = intermediate_steps
            return new_output

        return RunnableParallel(
            **{
                cypher_query_generation_chain.output_key: question_input_chain,
                semantic_scholar_chain.output_key: semantic_scholar_chain,
            }
        ) | RunnableLambda(_flatten_output)

    def _build_cypher_query_preprocessors_chain(self, config: GraphQAChainConfig) -> CypherQueryPreprocessorsChain:
        return CypherQueryPreprocessorsChain(
            cypher_query_preprocessors=config.cypher_query_preprocessors,
            return_intermediate_steps=config.return_intermediate_steps,
        )

    def _build_graph_chain(self, config: GraphQAChainConfig) -> GraphChain:
        return GraphChain(
            graph=config.graph, return_intermediate_steps=config.return_intermediate_steps, top_k=config.top_k
        )

    def _build_answer_generation_chain(
        self, config: GraphQAChainConfig, semantic_scholar_chain: Optional[SemanticScholarChain]
    ) -> CombinedQAChain | AnswerGenerationChain:
        if semantic_scholar_chain is not None:
            assert config.combined_answer_generation_prompt is not None
            return CombinedQAChain(
                llm=config.llm,
                combined_answer_generation_template=config.combined_answer_generation_prompt,
                rag_output_key=semantic_scholar_chain.output_key,
                return_intermediate_steps=config.return_intermediate_steps,
            )
        return AnswerGenerationChain(
            llm=config.llm,
            prompt_template=config.answer_generation_prompt,
            return_intermediate_steps=config.return_intermediate_steps,
        )

    def _subgraph_extractor_chain(self, config: GraphQAChainConfig) -> SubgraphExtractorChain:
        return SubgraphExtractorChain(
            llm=config.llm,
            graph=config.graph,
            return_intermediate_steps=config.return_intermediate_steps,
            subgraph_expansion=SubgraphExpansion(graph=config.graph),
            use_subgraph_expansion=config.use_subgraph_expansion,
        )

    def _build_output_chain(self, config: GraphQAChainConfig) -> GraphQAChainOutputPreparation:
        return GraphQAChainOutputPreparation(return_intermediate_steps=config.return_intermediate_steps)
