from typing import List, Any, Optional, Dict

from langchain.chains import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult


class CustomLLMChain(LLMChain):
    filled_prompt_template: str = None

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        self.filled_prompt_template = prompts[0].text
        callbacks = run_manager.get_child() if run_manager else None
        if isinstance(self.llm, BaseLanguageModel):
            result = self.llm.generate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **self.llm_kwargs,
            )
            return result
