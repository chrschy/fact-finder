from typing import Dict, Any, Optional, List

from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel

from fact_finder.tools.sub_graph_extractor import LLMSubGraphExtractor


class SubgraphExtractorChain(Chain):
    graph: Neo4jGraph
    subgraph_extractor: LLMSubGraphExtractor
    return_intermediate_steps: bool
    input_key: str = "cypher_query"  #: :meta private:
    output_key: str = "extracted_nodes"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"

    def __init__(self, llm: BaseLanguageModel, graph: Neo4jGraph, return_intermediate_steps: bool = True):
        subgraph_extractor = LLMSubGraphExtractor(llm)
        super().__init__(
            subgraph_extractor=subgraph_extractor,
            graph=graph,
            return_intermediate_steps=return_intermediate_steps,
        )

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        cypher_query = inputs[self.input_key]
        subgraph_cypher = self.subgraph_extractor(cypher_query)
        self._log_it("Subgraph Cypher:", _run_manager, subgraph_cypher)

        extracted_nodes = self._query_graph(subgraph_cypher)
        self._log_it("Extracted Nodes:", _run_manager, extracted_nodes)

        return self._prepare_chain_result(inputs, subgraph_cypher, extracted_nodes)

    def _query_graph(self, subgraph_cypher) -> List[Dict[str, Any]]:
        try:
            return self.graph.query(subgraph_cypher)
        except Exception as e:
            print(f"Sub Graph for {subgraph_cypher} could not be extracted due to {e}")
        return []

    def _log_it(self, text: str, _run_manager: CallbackManagerForChainRun, subgraph_cypher: str):
        _run_manager.on_text(text, end="\n", verbose=self.verbose)
        _run_manager.on_text(subgraph_cypher, color="green", end="\n", verbose=self.verbose)

    def _prepare_chain_result(
        self, inputs: Dict[str, Any], subgraph_cypher: str, extracted_nodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        chain_result = {
            self.output_key: extracted_nodes,
        }
        if self.return_intermediate_steps:
            intermediate_steps = inputs[self.intermediate_steps_key] + [{"subgraph_cypher": subgraph_cypher}]
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result
