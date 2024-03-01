from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel

from fact_finder.chains.custom_llm_chain import CustomLLMChain
from fact_finder.prompt_templates import CYPHER_QA_PROMPT

load_dotenv()


class QAChain(Chain):
    llm_chain: CustomLLMChain
    return_intermediate_steps: bool = True
    input_key: str = "graph_result"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"
    question_key: str = "question"
    filled_prompt_template_key: str = "qa_filled_prompt_template"

    def __init__(
        self,
        llm: BaseLanguageModel,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
    ):
        llm_chain = CustomLLMChain(llm=llm, prompt=CYPHER_QA_PROMPT)
        if exclude_types and include_types:
            raise ValueError("Either `exclude_types` or `include_types` " "can be provided, but not both")
        super().__init__(
            llm_chain=llm_chain,
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
        callbacks = _run_manager.get_child()

        graph_result = inputs[self.input_key]
        question = inputs[self.intermediate_steps_key][0][self.question_key]
        answer = self._run_qa_chain(callbacks, graph_result, question)

        self._log_it(_run_manager, answer)

        chain_result = {
            self.output_key: answer,
        }

        if self.return_intermediate_steps:
            intermediate_steps = inputs[self.intermediate_steps_key]
            intermediate_steps.append({self.output_key: answer})
            intermediate_steps.append({self.filled_prompt_template_key: self.llm_chain.filled_prompt_template})
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result

    def _run_qa_chain(self, callbacks, graph_result, question):
        final_result = self.llm_chain(
            {"question": question, "context": graph_result},
            callbacks=callbacks,
        )[self.llm_chain.output_key]
        return final_result

    def _log_it(self, run_manager, graph_result):
        run_manager.on_text("QA Chain Result:", end="\n", verbose=self.verbose)
        run_manager.on_text(str(graph_result), color="green", end="\n", verbose=self.verbose)
