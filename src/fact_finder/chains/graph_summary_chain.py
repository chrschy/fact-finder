from typing import Dict, Any, Optional, List

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from fact_finder.prompt_templates import SUBGRAPH_SUMMARY_PROMPT


class GraphSummaryChain(Chain):
    graph_summary_llm_chain: LLMChain
    return_intermediate_steps: bool = True
    input_key: str = "sub_graph"  #: :meta private:
    output_key: str = "summary"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def __init__(
        self,
        llm: BaseLanguageModel,
        graph_summary_template: PromptTemplate,
        return_intermediate_steps: bool = True,
    ):
        graph_summary_llm_chain = LLMChain(llm=llm, prompt=graph_summary_template)
        super().__init__(
            llm=llm,
            graph_summary_llm_chain=graph_summary_llm_chain,
            return_intermediate_steps=return_intermediate_steps,
        )

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        #todo add intermediate steps
        graph_triplets = inputs[self.input_key]

        summary = self.graph_summary_llm_chain(
            {"sub_graph": graph_triplets}, callbacks=run_manager.get_child(),
        )[self.graph_summary_llm_chain.output_key]

        result: Dict[str, Any] = {self.output_key: summary}
        if self.return_intermediate_steps and self.intermediate_steps_key in inputs:
            result[self.intermediate_steps_key] = inputs[self.intermediate_steps_key]
        return result