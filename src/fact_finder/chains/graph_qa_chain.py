from typing import Any, Dict, Iterable, List, Optional

from fact_finder.chains.answer_generation_chain import AnswerGenerationChain
from fact_finder.chains.cypher_query_generation_chain import CypherQueryGenerationChain
from fact_finder.chains.cypher_query_preprocessors_chain import (
    CypherQueryPreprocessorsChain,
)
from fact_finder.chains.graph_chain import GraphChain
from fact_finder.chains.subgraph_extractor_chain import SubgraphExtractorChain
from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import (
    CypherQueryPreprocessor,
)
from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSequence
from pydantic import BaseModel

from fact_finder.tools.subgraph_extension import SubgraphExpansion


class GraphQAChainOutput(BaseModel):
    question: str
    cypher_query: str
    graph_response: List[Dict[str, Any]]
    answer: str
    evidence_sub_graph: List[Dict[str, Any]]
    expanded_evidence_subgraph: List[Dict[str, Any]]


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

    def __init__(
        self,
        llm: BaseLanguageModel,
        graph: Neo4jGraph,
        cypher_prompt: BasePromptTemplate,
        answer_generation_prompt: BasePromptTemplate,
        cypher_query_preprocessors: List[CypherQueryPreprocessor],
        predicate_descriptions: List[Dict[str, str]] = [],
        schema_error_string: str = "SCHEMA_ERROR",
        return_intermediate_steps: bool = True,
    ):
        combined_chain = self._build_chain(
            llm,
            graph,
            cypher_prompt,
            answer_generation_prompt,
            cypher_query_preprocessors,
            predicate_descriptions,
            schema_error_string,
            return_intermediate_steps,
        )
        super().__init__(
            schema_error_string=schema_error_string,
            combined_chain=combined_chain,
            return_intermediate_steps=return_intermediate_steps,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, GraphQAChainOutput | List[Dict[str, Any]]]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        config = RunnableConfig(callbacks=_run_manager.get_child())
        chain_result = self.combined_chain.invoke(inputs[self.input_key], config=config)
        result = {self.output_key: chain_result["graph_qa_output"]}
        if self.return_intermediate_steps:
            result[self.intermediate_steps_key] = chain_result[self.intermediate_steps_key]
        return result

    def _build_chain(
        self,
        llm: BaseLanguageModel,
        graph: Neo4jGraph,
        cypher_prompt: BasePromptTemplate,
        answer_generation_prompt: BasePromptTemplate,
        cypher_query_preprocessors: List[CypherQueryPreprocessor],
        predicate_descriptions: List[Dict[str, str]] = [],
        schema_error_string: str = "SCHEMA_ERROR",
        return_intermediate_steps: bool = True,
    ):
        cypher_query_generation_chain = CypherQueryGenerationChain(
            llm=llm,
            graph_structured_schema=graph.get_structured_schema,
            prompt_template=cypher_prompt,
            predicate_descriptions=predicate_descriptions,
            return_intermediate_steps=return_intermediate_steps,
        )
        cypher_query_preprocessors_chain = CypherQueryPreprocessorsChain(
            cypher_query_preprocessors=cypher_query_preprocessors, return_intermediate_steps=return_intermediate_steps
        )
        graph_chain = GraphChain(graph=graph, return_intermediate_steps=return_intermediate_steps)
        answer_generation_chain = AnswerGenerationChain(
            llm=llm, prompt_template=answer_generation_prompt, return_intermediate_steps=return_intermediate_steps
        )
        subgraph_extractor_chain = SubgraphExtractorChain(
            llm=llm,
            graph=graph,
            return_intermediate_steps=return_intermediate_steps,
            subgraph_expansion=SubgraphExpansion(graph=graph),
            use_subgraph_expansion=True,
        )
        output_chain = GraphQAChainOutputPreparation(return_intermediate_steps=return_intermediate_steps)

        combined_chain = (
            cypher_query_preprocessors_chain
            | {
                subgraph_extractor_chain.output_key: subgraph_extractor_chain,
                answer_generation_chain.output_key: graph_chain | answer_generation_chain,
            }
            | output_chain
        )

        def route(inputs):
            if inputs[cypher_query_generation_chain.output_key].startswith(schema_error_string):
                return GraphQAChainEarlyStopping(schema_error_string=schema_error_string)
            return combined_chain

        return cypher_query_generation_chain | RunnableLambda(route)


class GraphQAChainEarlyStopping(Chain):
    schema_error_string: str
    return_intermediate_steps: bool = True
    question_key: str = "question"  #: :meta private:
    query_key: str = "cypher_query"  #: :meta private:
    output_key: str = "graph_qa_output"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return [self.question_key, self.query_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        answer = inputs[self.query_key][len(self.schema_error_string) :].lstrip(": ")
        result: Dict[str, Any] = {
            self.output_key: GraphQAChainOutput(
                question=inputs[self.question_key],
                cypher_query="",
                graph_response=[],
                answer=answer,
                evidence_sub_graph=[],
                expanded_evidence_subgraph=[],
            )
        }
        if self.return_intermediate_steps and self.intermediate_steps_key in inputs:
            result[self.intermediate_steps_key] = inputs[self.intermediate_steps_key]
        return result


class GraphQAChainOutputPreparation(Chain):
    return_intermediate_steps: bool = True
    answer_key: str = "answer"  #: :meta private:
    question_key: str = "question"  #: :meta private:
    query_key: str = "preprocessed_cypher_query"  #: :meta private:
    graph_key: str = "graph_result"  #: :meta private:
    evidence_key: str = "extracted_nodes"  #: :meta private:
    expanded_evidence_key: str = "expanded_nodes"  #: :meta private:
    output_key: str = "graph_qa_output"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return [self.answer_key, self.evidence_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        print("bla")
        result: Dict[str, Any] = {
            self.output_key: GraphQAChainOutput(
                question=inputs[self.answer_key][self.question_key],
                cypher_query=inputs[self.answer_key][self.query_key],
                graph_response=inputs[self.answer_key][self.graph_key],
                answer=inputs[self.answer_key][self.answer_key],
                evidence_sub_graph=inputs[self.evidence_key][self.evidence_key][self.evidence_key],
                expanded_evidence_subgraph=inputs[self.evidence_key][self.evidence_key][self.expanded_evidence_key],
            )
        }
        if self.return_intermediate_steps:
            intermediate_steps = list(self._merge_intermediate_steps(inputs))
            result[self.intermediate_steps_key] = intermediate_steps
        return result

    def _merge_intermediate_steps(self, inputs: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        i1 = iter(inputs[self.answer_key][self.intermediate_steps_key])
        i2 = iter(inputs[self.evidence_key][self.intermediate_steps_key])
        i2_store = None
        try:
            while True:
                i1_entry = next(i1)
                yield i1_entry
                i2_entry = next(i2)
                if i1_entry != i2_entry:
                    i2_store = i2_entry
                    break
        except StopIteration:
            pass
        yield from i1
        if i2_store:
            yield i2_store
        yield from i2
