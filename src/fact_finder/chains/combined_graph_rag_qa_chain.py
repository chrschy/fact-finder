from typing import Any, Dict, List

from fact_finder.chains.rag.text_search_qa_chain import TextSearchQAChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate


class CombinedQAChain(TextSearchQAChain):
    graph_result_key: str = "graph_result"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        return [self.question_key, self.graph_result_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def __init__(
        self,
        llm: BaseLanguageModel,
        combined_answer_generation_template: BasePromptTemplate,
        rag_output_key: str,
        return_intermediate_steps: bool = True,
    ):
        super().__init__(
            llm=llm,
            rag_answer_generation_template=combined_answer_generation_template,
            rag_output_key=rag_output_key,
            return_intermediate_steps=return_intermediate_steps,
        )

    def _generate_answer(self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun) -> str:
        return self.rag_answer_generation_llm_chain(
            {
                "abstracts": inputs[self.rag_output_key],
                "graph_answer": inputs[self.graph_result_key],
                "question": inputs[self.question_key],
            },
            callbacks=run_manager.get_child(),
        )[self.rag_answer_generation_llm_chain.output_key]
