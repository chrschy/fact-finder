from unittest.mock import MagicMock

from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation, LLMResult


def build_llm_mock(output: str) -> BaseLanguageModel:
    llm = MagicMock(spec=BaseLanguageModel)
    llm.generate_prompt = MagicMock()
    llm.generate_prompt.return_value = LLMResult(generations=[[Generation(text=output)]])
    return llm
