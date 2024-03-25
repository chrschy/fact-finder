from typing import Any, Dict, Iterable, List, Optional

from fact_finder.chains.graph_qa_chain.output import GraphQAChainOutput
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun


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
