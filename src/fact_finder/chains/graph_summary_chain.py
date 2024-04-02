from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from fact_finder.prompt_templates import SUBGRAPH_SUMMARY_PROMPT
from fact_finder.utils import fill_prompt_template


class GraphSummaryChain(Chain):
    graph_summary_llm_chain: LLMChain
    return_intermediate_steps: bool = True
    input_key: str = "sub_graph"  #: :meta private:
    output_key: str = "summary"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"  #: :meta private:
    filled_prompt: str = ""

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
        graph_triplets = inputs[self.input_key]
        inputs = {"sub_graph": graph_triplets}
        answer = self.graph_summary_llm_chain(
            inputs=inputs,
            callbacks=run_manager.get_child(),
        )[self.graph_summary_llm_chain.output_key]
        self.filled_prompt = fill_prompt_template(inputs=inputs, llm_chain=self.graph_summary_llm_chain)
        return self._prepare_chain_result(inputs, answer)

    def _prepare_chain_result(self, inputs, answer):
        chain_result: Dict[str, Any] = {self.output_key: answer}
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            intermediate_steps += [
                {"question": inputs[self.input_key]},
                {self.output_key: answer},
                {f"{self.__class__.__name__}_filled_prompt": self.filled_prompt},
            ]
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result
