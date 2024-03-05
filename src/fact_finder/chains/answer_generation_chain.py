from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate


class AnswerGenerationChain(Chain):
    llm_chain: LLMChain
    return_intermediate_steps: bool
    question_key: str = "question"  #: :meta private:
    graph_result_key: str = "graph_result"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"
    filled_prompt_template_key: str = "qa_filled_prompt_template"

    def __init__(
        self, llm: BaseLanguageModel, prompt_template: BasePromptTemplate, return_intermediate_steps: bool = True
    ):
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        super().__init__(llm_chain=llm_chain, return_intermediate_steps=return_intermediate_steps)

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return [self.question_key, self.graph_result_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        graph_result = inputs[self.graph_result_key]
        question = inputs[self.question_key]
        answer = self._run_qa_chain(graph_result, question, _run_manager)
        return self._prepare_chain_result(inputs, answer)

    def _run_qa_chain(
        self, graph_result: List[Dict[str, Any]], question: str, run_manager: CallbackManagerForChainRun
    ) -> str:
        final_result = self.llm_chain(
            {"question": question, "context": graph_result},
            callbacks=run_manager.get_child(),
        )[self.llm_chain.output_key]
        self._log_it(run_manager, final_result)
        return final_result

    def _log_it(self, run_manager, graph_result):
        run_manager.on_text("QA Chain Result:", end="\n", verbose=self.verbose)
        run_manager.on_text(str(graph_result), color="green", end="\n", verbose=self.verbose)

    def _prepare_chain_result(self, inputs: Dict[str, Any], answer: str) -> Dict[str, Any]:
        chain_result = {
            self.output_key: answer,
        }
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, []) + [{self.output_key: answer}]
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result
