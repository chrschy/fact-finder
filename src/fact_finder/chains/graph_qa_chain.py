from typing import Any, Dict, Iterable, List, Optional

from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableSequence
from pydantic import BaseModel

from fact_finder.chains.answer_generation_chain import AnswerGenerationChain
from fact_finder.chains.cypher_query_generation_chain import CypherQueryGenerationChain
from fact_finder.chains.cypher_query_preprocessors_chain import CypherQueryPreprocessorsChain
from fact_finder.chains.graph_chain import GraphChain
from fact_finder.chains.subgraph_extractor_chain import SubgraphExtractorChain
from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor


class GraphQAChainOutput(BaseModel):
    question: str
    cypher_query: str
    graph_response: List[Dict[str, Any]]
    answer: str
    evidence_sub_graph: List[Dict[str, Any]]


class GraphQAChain(Chain):
    combined_chain: RunnableSequence
    return_intermediate_steps: bool
    input_key: str = "question"  #: :meta private:
    output_key: str = "graph_qa_output"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"

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
        return_intermediate_steps: bool = True,
    ):
        cypher_query_generation_chain = CypherQueryGenerationChain(
            llm=llm,
            graph_structured_schema=graph.get_structured_schema,
            prompt_template=cypher_prompt,
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
            llm=llm, graph=graph, return_intermediate_steps=return_intermediate_steps
        )
        output_chain = GraphQAChainOutputPreparation(return_intermediate_steps=return_intermediate_steps)
        combined_chain = (
            cypher_query_generation_chain
            | cypher_query_preprocessors_chain
            | {
                subgraph_extractor_chain.output_key: subgraph_extractor_chain,
                answer_generation_chain.output_key: graph_chain | answer_generation_chain,
            }
            | output_chain
        )
        super().__init__(combined_chain=combined_chain, return_intermediate_steps=return_intermediate_steps)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, GraphQAChainOutput | List[Dict[str, Any]]]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        config = RunnableConfig(callbacks=_run_manager)
        chain_result = self.combined_chain.invoke(inputs[self.input_key], config=config)
        result = {self.output_key: chain_result["graph_qa_output"]}
        if self.return_intermediate_steps:
            result[self.intermediate_steps_key] = chain_result[self.intermediate_steps_key]
        return result


class GraphQAChainOutputPreparation(Chain):
    return_intermediate_steps: bool = True
    answer_key: str = "answer"  #: :meta private:
    question_key: str = "question"  #: :meta private:
    query_key: str = "cypher_query"  #: :meta private:
    graph_key: str = "graph_result"  #: :meta private:
    evidence_key: str = "extracted_nodes"  #: :meta private:
    output_key: str = "graph_qa_output"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"

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
        result: Dict[str, Any] = {
            self.output_key: GraphQAChainOutput(
                question=inputs[self.answer_key][self.question_key],
                cypher_query=inputs[self.answer_key][self.query_key],
                graph_response=inputs[self.answer_key][self.graph_key],
                answer=inputs[self.answer_key][self.answer_key],
                evidence_sub_graph=inputs[self.evidence_key][self.evidence_key],
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
