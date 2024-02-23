from typing import List
from fact_finder.prompt_templates import LLM_PROMPT
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.prompt import PromptTemplate


def build_chain(model: BaseLanguageModel, args: List[str] = []) -> Chain:
    prompt_template = _get_llm_prompt_template()
    return LLMChain(llm=model, prompt=prompt_template, verbose=True)


def _get_llm_prompt_template() -> PromptTemplate:
    return LLM_PROMPT
