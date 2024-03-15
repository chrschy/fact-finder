from typing import Dict, Any, Optional, List

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from fact_finder.tools.semantic_scholar_search_api_wrapper import SemanticScholarSearchApiWrapper


class TextSearchQAChain(Chain):
    semantic_scholar_search: SemanticScholarSearchApiWrapper
    keyword_generation_llm_chain: LLMChain
    rag_answer_generation_llm_chain: LLMChain
    return_intermediate_steps: bool = True
    input_key: str = "question"  #: :meta private:
    output_key: str = "rag_output"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def __init__(
        self,
        semantic_scholar_search: SemanticScholarSearchApiWrapper,
        llm: BaseLanguageModel,
        keyword_prompt_template: PromptTemplate,
        rag_answer_generation_template: PromptTemplate,
        return_intermediate_steps: bool = True,
    ):
        keyword_generation_llm_chain = LLMChain(llm=llm, prompt=keyword_prompt_template)
        rag_answer_generation_llm_chain = LLMChain(llm=llm, prompt=rag_answer_generation_template)
        super().__init__(
            semantic_scholar_search=semantic_scholar_search,
            llm=llm,
            keyword_generation_llm_chain=keyword_generation_llm_chain,
            rag_answer_generation_llm_chain=rag_answer_generation_llm_chain,
            return_intermediate_steps=return_intermediate_steps,
        )

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        #todo add intermediate steps
        user_question = inputs[self.input_key]
        keywords = self.keyword_generation_llm_chain(
            {"question": user_question},
            callbacks=run_manager.get_child(),
        )[self.keyword_generation_llm_chain.output_key]

        rag_inputs = self.semantic_scholar_search.search_by_abstracts(keywords=keywords)
        rag_inputs = "\n\n".join(rag_inputs)

        answer = self.rag_answer_generation_llm_chain(
            {"context": rag_inputs, "question": user_question},
            callbacks=run_manager.get_child(),
        )[self.rag_answer_generation_llm_chain.output_key]

        result: Dict[str, Any] = {self.output_key: answer}
        if self.return_intermediate_steps and self.intermediate_steps_key in inputs:
            result[self.intermediate_steps_key] = inputs[self.intermediate_steps_key]
        return result
