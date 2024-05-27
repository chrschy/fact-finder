from typing import Any, Dict, List, Optional, Tuple

from fact_finder.tools.sub_graph_extractor import LLMSubGraphExtractor
from fact_finder.tools.subgraph_extension import SubgraphExpansion
from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel


class SubgraphExtractorChain(Chain):
    graph: Neo4jGraph
    subgraph_extractor: LLMSubGraphExtractor
    subgraph_expansion: SubgraphExpansion
    use_subgraph_expansion: bool
    return_intermediate_steps: bool
    input_key: str = "preprocessed_cypher_query"  #: :meta private:
    output_key: str = "extracted_nodes"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"

    def __init__(
        self,
        llm: BaseLanguageModel,
        graph: Neo4jGraph,
        subgraph_expansion: SubgraphExpansion,
        use_subgraph_expansion: bool,
        return_intermediate_steps: bool = True,
    ):
        subgraph_extractor = LLMSubGraphExtractor(llm)
        super().__init__(
            subgraph_extractor=subgraph_extractor,
            graph=graph,
            subgraph_expansion=subgraph_expansion,
            use_subgraph_expansion=use_subgraph_expansion,
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
        try:
            subgraph_cypher, extracted_nodes, expanded_nodes = self._try_generating_subgraph(
                inputs[self.input_key], _run_manager
            )
        except Exception as e:
            self._log_it(f"Error when creating subgraph cypher!", _run_manager, e)
            subgraph_cypher = subgraph_cypher if "subgraph_cypher" in locals() else ""
            extracted_nodes = extracted_nodes if "extracted_nodes" in locals() else []
            expanded_nodes = []
        return self._prepare_chain_result(inputs, subgraph_cypher, extracted_nodes, expanded_nodes)

    def _try_generating_subgraph(
        self, cypher_query: str, _run_manager: CallbackManagerForChainRun
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        subgraph_cypher = self.subgraph_extractor(cypher_query)
        self._log_it("Subgraph Cypher:", _run_manager, subgraph_cypher)

        extracted_nodes = self.graph.query(subgraph_cypher)
        expanded_nodes = self.subgraph_expansion.expand(nodes=extracted_nodes) if self.use_subgraph_expansion else []
        self._log_it("Extracted Nodes:", _run_manager, extracted_nodes)

        return subgraph_cypher, extracted_nodes, expanded_nodes

    def _log_it(self, text: str, _run_manager: CallbackManagerForChainRun, entity: Any):
        _run_manager.on_text(text, end="\n", verbose=self.verbose)
        _run_manager.on_text(entity, color="green", end="\n", verbose=self.verbose)

    def _prepare_chain_result(
        self,
        inputs: Dict[str, Any],
        subgraph_cypher: str,
        extracted_nodes: List[Dict[str, Any]],
        expanded_nodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        chain_result = {
            self.output_key: {"extracted_nodes": extracted_nodes, "expanded_nodes": expanded_nodes},
        }
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, []) + [{"subgraph_cypher": subgraph_cypher}]
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result
