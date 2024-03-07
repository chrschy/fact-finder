from typing import Any, Dict, List, Optional, Tuple

from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import (
    CypherQueryPreprocessor,
)
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun


class CypherQueryPreprocessorsChain(Chain):
    cypher_query_preprocessors: List[CypherQueryPreprocessor]
    return_intermediate_steps: bool = True
    input_key: str = "cypher_query"  #: :meta private:
    output_key: str = "preprocessed_cypher_query"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"  #: :meta private:

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
        preprocessed_cypher, intermediate_steps = self._run_preprocessors(_run_manager, generated_cypher)
        return self._prepare_chain_result(inputs, preprocessed_cypher, intermediate_steps)

    def _run_preprocessors(
        self, _run_manager: CallbackManagerForChainRun, generated_cypher: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        intermediate_steps = []
        for processor in self.cypher_query_preprocessors:
            generated_cypher = processor(generated_cypher)
            intermediate_steps.append({type(processor).__name__: generated_cypher})
        self._log_it(_run_manager, generated_cypher)
        return generated_cypher, intermediate_steps

    def _log_it(self, _run_manager: CallbackManagerForChainRun, generated_cypher: str):
        _run_manager.on_text("Preprocessed Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(generated_cypher, color="green", end="\n", verbose=self.verbose)

    def _prepare_chain_result(
        self, inputs: Dict[str, Any], preprocessed_cypher: str, intermediate_steps: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        chain_result: Dict[str, Any] = {
            self.output_key: preprocessed_cypher,
        }
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, []) + intermediate_steps
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result
