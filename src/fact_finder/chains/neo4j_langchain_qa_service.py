from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.graph_qa.cypher import construct_schema, extract_cypher
from langchain.chains.llm import LLMChain
from langchain_community.graphs.graph_store import GraphStore
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field

from fact_finder.chains.custom_llm_chain import CustomLLMChain
from fact_finder.chains.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor
from fact_finder.tools.sub_graph_extractor import LLMSubGraphExtractor

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


class Neo4JLangchainQAService(Chain):
    """Based on langchain.chains.GraphCypherQAChain:
    Chain for question-answering against a graph by generating Cypher statements.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    graph: GraphStore = Field(exclude=True)
    cypher_generation_chain: CustomLLMChain
    qa_chain: CustomLLMChain
    graph_schema: str
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    top_k: int = 100
    llm_subgraph_extractor: LLMSubGraphExtractor
    """Number of results to return from the query"""
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the graph directly."""
    cypher_query_preprocessors: List[CypherQueryPreprocessor] = []
    """Optional cypher validation/preprocessing tools"""
    schema_error_string: Optional[str] = "SCHEMA_ERROR"
    """Optional string to be generated at the start of the cypher query to indicate an error."""
    n_predicate_descriptions: int = 0
    """How many relationship descriptions to include into the cypher generation prompt."""

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        return "graph_cypher_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        qa_prompt: BasePromptTemplate,
        cypher_prompt: BasePromptTemplate,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
        cypher_query_preprocessors: List[CypherQueryPreprocessor] = [],
        schema_error_string: Optional[str] = "SCHEMA_ERROR",
        **kwargs: Any,
    ) -> Neo4JLangchainQAService:
        """Initialize from LLM."""
        # QA
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        # Cypher
        cypher_generation_chain = LLMChain(llm=llm, prompt=cypher_prompt)
        if exclude_types and include_types:
            raise ValueError("Either `exclude_types` or `include_types` " "can be provided, but not both")
        graph_schema = construct_schema(kwargs["graph"].get_structured_schema, include_types, exclude_types)
        # Extractor
        llm_subgraph_extractor = LLMSubGraphExtractor(model=llm)
        # Neo4JLangchainQAService __init__()
        return cls(
            graph_schema=graph_schema,
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
            cypher_query_preprocessors=cypher_query_preprocessors,
            schema_error_string=schema_error_string,
            llm_subgraph_extractor=llm_subgraph_extractor,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher statement, use it to look up in db and answer question."""
        # logging
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        question = inputs[self.input_key]

        intermediate_steps: List = []

        generated_cypher = self._generate_cypher(callbacks, question, _run_manager)

        if self.schema_error_string is not None and generated_cypher.startswith(self.schema_error_string):
            return {self.output_key: generated_cypher}
            # is returned by the class and given to "meta-preprocessor" who checks for error
        generated_cypher = self._run_preprocessors(_run_manager, generated_cypher, intermediate_steps)

        # Graph QA Chain
        graph_result = self._query_graph(generated_cypher, intermediate_steps, run_manager)

        # QA Chain
        chain_result = self._run_qa_chain(callbacks, graph_result, question)

        # subgraph extractor
        self._run_subgraph_extractor(chain_result, generated_cypher)

        # this needs to happen in every chain
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result

    def _run_subgraph_extractor(self, chain_result, generated_cypher):
        subgraph_cypher = self.llm_subgraph_extractor(generated_cypher)
        try:
            chain_result["sub_graph"] = self.graph.query(subgraph_cypher)
        except Exception as e:
            chain_result["sub_graph"] = []
            print(f"Sub Graph could not be extracted due to {e}")

    def _run_qa_chain(self, callbacks, graph_result, question):
        final_result = self.qa_chain(
            {"question": question, "context": graph_result},
            callbacks=callbacks,
        )[self.qa_chain.output_key]
        chain_result: Dict[str, Any] = {self.output_key: final_result}
        return chain_result

    def _query_graph(self, generated_cypher, intermediate_steps, run_manager):
        # Retrieve and limit the number of results
        # Generated Cypher be null if query corrector identifies invalid schema
        if generated_cypher:
            graph_result = self.graph.query(generated_cypher)[: self.top_k]
        else:
            graph_result = []
        intermediate_steps.append({"graph_result": graph_result})
        run_manager.on_text("Graph Result:", end="\n", verbose=self.verbose)
        run_manager.on_text(str(graph_result), color="green", end="\n", verbose=self.verbose)
        return graph_result

    def _run_preprocessors(self, _run_manager, generated_cypher, intermediate_steps):
        # Correct Cypher query if enabled
        for processor in self.cypher_query_preprocessors:
            generated_cypher = processor(generated_cypher)
        # logging for cypher generator
        _run_manager.on_text("Pre-processed Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(generated_cypher, color="green", end="\n", verbose=self.verbose)
        intermediate_steps.append({"pre_processed_cypher_query": generated_cypher})
        return generated_cypher
