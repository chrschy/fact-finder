from typing import Any, Dict, List, Optional

from fact_finder.tools.semantic_scholar_search_api_wrapper import (
    SemanticScholarSearchApiWrapper,
)
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate


class SemanticScholarChain(Chain):
    semantic_scholar_search: SemanticScholarSearchApiWrapper
    keyword_generation_llm_chain: LLMChain
    return_intermediate_steps: bool = True
    question_key: str = "question"  #: :meta private:
    output_key: str = "semantic_scholar_result"  #: :meta private:
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
        keyword_prompt_template: BasePromptTemplate,
        return_intermediate_steps: bool = True,
    ):
        keyword_generation_llm_chain = LLMChain(llm=llm, prompt=keyword_prompt_template)
        super().__init__(
            semantic_scholar_search=semantic_scholar_search,
            keyword_generation_llm_chain=keyword_generation_llm_chain,
            return_intermediate_steps=return_intermediate_steps,
        )

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        keywords = self._generate_search_keywords_for_question(inputs, run_manager)
        semantic_scholar_result = self._search_semantic_scholar(keywords)
        return self._build_result(inputs, keywords, semantic_scholar_result)

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

    def _build_result(self, inputs: Dict[str, Any], keywords: str, semantic_scholar_result: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {self.output_key: semantic_scholar_result}
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            intermediate_steps.append(("search_keywords", keywords))
            intermediate_steps.append(("semantic_scholar_result", semantic_scholar_result))
            result[self.intermediate_steps_key] = intermediate_steps
        return result
