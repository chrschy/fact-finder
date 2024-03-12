from typing import List

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.prompt import PromptTemplate

from fact_finder.chains.rag.text_search_qa_chain import TextSearchQAChain
from fact_finder.prompt_templates import KEYWORD_PROMPT, LLM_PROMPT, RAG_PROMPT
from fact_finder.tools.semantic_scholar_search_api_wrapper import SemanticScholarSearchApiWrapper


def build_chain(model: BaseLanguageModel, args: List[str] = []) -> Chain:
    prompt_template = _get_llm_prompt_template()
    return LLMChain(llm=model, prompt=prompt_template, verbose=True)


def build_rag_chain(model: BaseLanguageModel, args: List[str] = []) -> Chain:
    return TextSearchQAChain(
        semantic_scholar_search=SemanticScholarSearchApiWrapper(),
        llm=model,
        keyword_prompt_template=KEYWORD_PROMPT,
        rag_answer_generation_template=RAG_PROMPT,
    )


def _get_llm_prompt_template() -> PromptTemplate:
    return LLM_PROMPT
