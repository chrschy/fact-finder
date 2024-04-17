from typing import Any, Dict, List

from fact_finder.chains.rag.text_search_qa_chain import TextSearchQAChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate

from fact_finder.utils import fill_prompt_template


class CombinedQAChain(TextSearchQAChain):
    cypher_query_key: str = "preprocessed_cypher_query"  #: :meta private:
    graph_result_key: str = "graph_result"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        return [self.question_key, self.graph_result_key, self.cypher_query_key]

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
        inputs = self._prepare_chain_input(inputs)
        result = self.rag_answer_generation_llm_chain(
            inputs=inputs,
            callbacks=run_manager.get_child(),
        )[self.rag_answer_generation_llm_chain.output_key]
        return result

    def _prepare_chain_input(self, inputs: Dict[str, Any]):
        return {
            "abstracts": inputs["semantic_scholar_result"],
            "cypher_query": inputs["cypher_query"],
            "graph_answer": inputs["graph_result"],
            "question": inputs["question"],
        }
