from typing import Any, Dict, List, Optional

from fact_finder.chains.graph_qa_chain.output import GraphQAChainOutput
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun


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
