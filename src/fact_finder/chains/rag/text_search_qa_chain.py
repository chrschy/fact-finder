from typing import Any, Dict, List, Optional

from fact_finder.tools.semantic_scholar_search_api_wrapper import (
    SemanticScholarSearchApiWrapper,
)
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate


class TextSearchQAChain(Chain):
    semantic_scholar_search: SemanticScholarSearchApiWrapper
    keyword_generation_llm_chain: LLMChain
    rag_answer_generation_llm_chain: LLMChain
    return_intermediate_steps: bool = True
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
            keyword_generation_llm_chain=keyword_generation_llm_chain,
            rag_answer_generation_llm_chain=rag_answer_generation_llm_chain,
            return_intermediate_steps=return_intermediate_steps,
        )

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        keywords = self._generate_search_keywords_for_question(inputs, run_manager)
        rag_inputs = self._search_semantic_scholar(keywords)
        answer = self._generate_answer(inputs, rag_inputs, run_manager)
        return self._build_result(inputs, keywords, rag_inputs, answer)

    def _generate_search_keywords_for_question(
        self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun
    ) -> str:
        return self.keyword_generation_llm_chain(
            {"question": inputs[self.question_key]},
            callbacks=run_manager.get_child(),
        )[self.keyword_generation_llm_chain.output_key]

    def _search_semantic_scholar(self, keywords: str) -> str:
        search_result = self.semantic_scholar_search.search_by_abstracts(keywords=keywords)
        return "\n\n".join(search_result)

    def _generate_answer(self, inputs: Dict[str, Any], rag_inputs: str, run_manager: CallbackManagerForChainRun) -> str:
        return self.rag_answer_generation_llm_chain(
            {"context": rag_inputs, "question": inputs[self.question_key]},
            callbacks=run_manager.get_child(),
        )[self.rag_answer_generation_llm_chain.output_key]

    def _build_result(self, inputs: Dict[str, Any], keywords: str, rag_inputs: str, answer: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {self.output_key: answer}
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            intermediate_steps.append(("search_keywords", keywords))
            intermediate_steps.append(("semantic_scholar_result", rag_inputs))
            result[self.intermediate_steps_key] = intermediate_steps
        return result
