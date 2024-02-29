from typing import Dict, Any, Optional, List

from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForChainRun


class GraphChain(Chain):
    graph: Neo4jGraph
    return_intermediate_steps: bool = True
    top_k: int = 20
    input_key: str = "preprocessed_cypher_query"  #: :meta private:
    output_key: str = "graph_result"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"

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
        generated_cypher = inputs[self.input_key]
        intermediate_steps = inputs[self.intermediate_steps_key]
        graph_result = self._query_graph(generated_cypher)
        intermediate_steps.append({self.output_key: graph_result})
        chain_result = {
            self.output_key: graph_result,
        }
        self._log_it(_run_manager, graph_result)
        if self.return_intermediate_steps:
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result

    def _query_graph(self, generated_cypher):
        if generated_cypher:
            graph_result = self.graph.query(generated_cypher)[: self.top_k]
        else:
            graph_result = []
        return graph_result

    def _log_it(self, run_manager, graph_result):
        run_manager.on_text("Graph Result:", end="\n", verbose=self.verbose)
        run_manager.on_text(str(graph_result), color="green", end="\n", verbose=self.verbose)
