from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain.chains.base import Chain
from langchain.chains.graph_qa.cypher import construct_schema, extract_cypher
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain_community.graphs.graph_store import GraphStore
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field

from fact_finder.qa_service.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor
from fact_finder.tools.sub_graph_extractor import LLMSubGraphExtractor
from fact_finder.qa_service.qa_service import QAService

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


class Neo4JLangchainQAService(QAService, Chain):
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
    cypher_generation_chain: LLMChain
    qa_chain: LLMChain
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

    def search(self, user_query: str) -> str:
        return self._call(inputs={self.input_key: user_query})["result"]

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    @property
    def _chain_type(self) -> str:
        return "graph_cypher_chain"

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        *,
        qa_prompt: Optional[BasePromptTemplate] = None,
        cypher_prompt: Optional[BasePromptTemplate] = None,
        cypher_llm: Optional[BaseLanguageModel] = None,
        qa_llm: Optional[BaseLanguageModel] = None,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
        cypher_query_preprocessors: List[CypherQueryPreprocessor] = [],
        qa_llm_kwargs: Optional[Dict[str, Any]] = None,
        cypher_llm_kwargs: Optional[Dict[str, Any]] = None,
        schema_error_string: Optional[str] = "SCHEMA_ERROR",
        **kwargs: Any,
    ) -> Neo4JLangchainQAService:
        """Initialize from LLM."""

        if not cypher_llm and not llm:
            raise ValueError("Either `llm` or `cypher_llm` parameters must be provided")
        if not qa_llm and not llm:
            raise ValueError("Either `llm` or `qa_llm` parameters must be provided")
        if cypher_llm and qa_llm and llm:
            raise ValueError(
                "You can specify up to two of 'cypher_llm', 'qa_llm'" ", and 'llm', but not all three simultaneously."
            )
        if cypher_prompt and cypher_llm_kwargs:
            raise ValueError(
                "Specifying cypher_prompt and cypher_llm_kwargs together is"
                " not allowed. Please pass prompt via cypher_llm_kwargs."
            )
        if qa_prompt and qa_llm_kwargs:
            raise ValueError(
                "Specifying qa_prompt and qa_llm_kwargs together is"
                " not allowed. Please pass prompt via qa_llm_kwargs."
            )
        use_qa_llm_kwargs = qa_llm_kwargs if qa_llm_kwargs is not None else {}
        use_cypher_llm_kwargs = cypher_llm_kwargs if cypher_llm_kwargs is not None else {}
        if "prompt" not in use_qa_llm_kwargs:
            use_qa_llm_kwargs["prompt"] = qa_prompt if qa_prompt is not None else CYPHER_QA_PROMPT
        if "prompt" not in use_cypher_llm_kwargs:
            use_cypher_llm_kwargs["prompt"] = cypher_prompt if cypher_prompt is not None else CYPHER_GENERATION_PROMPT

        qa_chain = LLMChain(llm=qa_llm or llm, **use_qa_llm_kwargs)

        cypher_generation_chain = LLMChain(llm=cypher_llm or llm, **use_cypher_llm_kwargs)

        if exclude_types and include_types:
            raise ValueError("Either `exclude_types` or `include_types` " "can be provided, but not both")

        graph_schema = construct_schema(kwargs["graph"].get_structured_schema, include_types, exclude_types)

        llm_subgraph_extractor = LLMSubGraphExtractor(model=qa_llm or llm)

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
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]

        logging.info("started")

        intermediate_steps: List = []

        predicate_descriptions = self._construct_predicate_descriptions(how_many=self.n_predicate_descriptions)

        generated_cypher = self.cypher_generation_chain(
            {"question": question, "schema": self.graph_schema, "predicate_descriptions": predicate_descriptions},
            callbacks=callbacks,
        )[self.cypher_generation_chain.output_key]

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher)

        if self.schema_error_string is not None and generated_cypher.startswith(self.schema_error_string):
            return {self.output_key: generated_cypher}

        # Correct Cypher query if enabled
        for processor in self.cypher_query_preprocessors:
            generated_cypher = processor(generated_cypher)

        _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(generated_cypher, color="green", end="\n", verbose=self.verbose)

        intermediate_steps.append({"query": generated_cypher})

        # Retrieve and limit the number of results
        # Generated Cypher be null if query corrector identifies invalid schema
        if generated_cypher:
            context = self.graph.query(generated_cypher)[: self.top_k]
        else:
            context = []

        if self.return_direct:
            final_result = context
        else:
            _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
            _run_manager.on_text(str(context), color="green", end="\n", verbose=self.verbose)
            intermediate_steps.append({"context": context})
            result = self.qa_chain(
                {"question": question, "context": context},
                callbacks=callbacks,
            )
            final_result = result[self.qa_chain.output_key]
        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        subgraph_cypher = self.llm_subgraph_extractor(generated_cypher)
        try:
            chain_result["sub_graph"] = self.graph.query(subgraph_cypher)
        except Exception as e:
            chain_result["sub_graph"] = []
            print(f"Sub Graph could not be extracted due to {e}")

        return chain_result

    def _construct_predicate_descriptions(self, how_many: int) -> str:
        if how_many == 0:
            return ""
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        path += "/../../../data/predicate_descriptions.csv"
        df = pd.read_csv(path, sep=";")
        result = ["Here are some descriptions to the most common relationships:"]
        for index, row in df.iterrows():
            if index == how_many - 1:
                break
            row_as_text = f"({row['subject']})-[{row['predicate']}]->({row['object']}): {row['definition']}"
            result.append(row_as_text)
        result = "\n".join(result)
        return result
