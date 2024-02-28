from typing import Dict, Any, Optional, List

from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForChainRun


class GraphChain(Chain):
    graph: Neo4jGraph
    return_intermediate_steps: bool
    input_key: str = "cypher_query"  #: :meta private:
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
        pass

    def _query_graph(self, generated_cypher, intermediate_steps, run_manager):
        # Retrieve and limit the number of results
        # Generated Cypher be null if query corrector identifies invalid schema
        if generated_cypher:
            graph_result = self.graph.query(generated_cypher)[: self.top_k]
        else:
            graph_result = []
        intermediate_steps.append({"graph_result": graph_result})

        return graph_result

    def log_it(self, run_manager, graph_result):
        run_manager.on_text("Graph Result:", end="\n", verbose=self.verbose)
        run_manager.on_text(str(graph_result), color="green", end="\n", verbose=self.verbose)
