from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate

from fact_finder.utils import fill_prompt_template


class TextSearchQAChain(Chain):
    rag_answer_generation_llm_chain: LLMChain
    return_intermediate_steps: bool = True
    rag_output_key: str
    question_key: str = "question"  #: :meta private:
    output_key: str = "rag_output"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def __init__(
        self,
        llm: BaseLanguageModel,
        rag_answer_generation_template: BasePromptTemplate,
        rag_output_key: str,
        return_intermediate_steps: bool = True,
    ):
        rag_answer_generation_llm_chain = LLMChain(llm=llm, prompt=rag_answer_generation_template)
        super().__init__(
            rag_answer_generation_llm_chain=rag_answer_generation_llm_chain,
            rag_output_key=rag_output_key,
            return_intermediate_steps=return_intermediate_steps,
        )

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        answer = self._generate_answer(inputs, run_manager)
        return self._prepare_chain_result(inputs, answer)

    def _generate_answer(self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun) -> str:
        result = self.rag_answer_generation_llm_chain(
            inputs=self._prepare_chain_input(inputs),
            callbacks=run_manager.get_child(),
        )[self.rag_answer_generation_llm_chain.output_key]
        return result

    def _prepare_chain_result(self, inputs: Dict[str, Any], answer: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {self.output_key: answer}
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            intermediate_steps.append(("rag_answer", answer))
            filled_prompt = fill_prompt_template(
                inputs=self._prepare_chain_input(inputs),
                llm_chain=self.rag_answer_generation_llm_chain,
            )
            intermediate_steps.append({f"{self.__class__.__name__}_filled_prompt": filled_prompt})
            result[self.intermediate_steps_key] = intermediate_steps
        return result

    def _prepare_chain_input(self, inputs: Dict[str, Any]):
        return {"context": inputs["semantic_scholar_result"], "question": inputs["question"]}
