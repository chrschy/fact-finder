from typing import Dict, Any, Optional, List

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun

from fact_finder.chains.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor


class CypherQueryPreprocessorsChain(Chain):
    cypher_query_preprocessors: List[CypherQueryPreprocessor]
    return_intermediate_steps: bool
    input_key: str = "cypher_query"  #: :meta private:
    output_key: str = "preprocessed_cypher_query"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"

    # todo having an __init__() throws a pydantic error
    """
    we can still initialize the chain with e.g.
    chain = PreprocessorsChain(return_intermediate_steps=True, cypher_query_preprocessors=preprocessors)
    but this is not as readable as having a constructor
    
    def __init__(
        self, cypher_query_preprocessors: List[CypherQueryPreprocessor] = [], return_intermediate_steps: bool = True
    ):
        self.cypher_query_preprocessors = cypher_query_preprocessors
        self.return_intermediate_steps = return_intermediate_steps
        super().__init__(
            cypher_query_preprocessors=cypher_query_preprocessors,
            return_intermediate_steps=return_intermediate_steps,
        )
    """

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
        preprocessed_cypher = self._run_preprocessors(run_manager, generated_cypher, intermediate_steps)
        chain_result = {
            self.output_key: preprocessed_cypher,
        }
        if self.return_intermediate_steps:
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result

    def _run_preprocessors(
        self, _run_manager: CallbackManagerForChainRun, generated_cypher: str, intermediate_steps: List
    ):
        intermediate_steps.append({})
        for processor in self.cypher_query_preprocessors:
            generated_cypher = processor(generated_cypher)
            self._update_intermediate_steps(generated_cypher, intermediate_steps, processor)
        self._log_it(_run_manager, generated_cypher, intermediate_steps)
        return generated_cypher

    def _update_intermediate_steps(
        self, generated_cypher: str, intermediate_steps: List, processor: CypherQueryPreprocessor
    ):
        name_of_preprocessor = type(processor).__name__
        intermediate_steps[-1][name_of_preprocessor] = generated_cypher

    def _log_it(self, _run_manager: CallbackManagerForChainRun, generated_cypher: str, intermediate_steps: List):
        _run_manager.on_text("Preprocessed Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(generated_cypher, color="green", end="\n", verbose=self.verbose)
