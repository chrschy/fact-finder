from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel

from fact_finder.prompt_templates import SUBGRAPH_PREPROCESSOR_PROMPT
from fact_finder.qa_service.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor


class ReturnSubgraphWithLLMPreprocessor(CypherQueryPreprocessor):

    def __init__(self, model: BaseLanguageModel):
        self.llm_chain = LLMChain(llm=model, prompt=SUBGRAPH_PREPROCESSOR_PROMPT)

    def __call__(self, cypher_query: str) -> str:
        result = self.llm_chain(cypher_query)
        return result["text"]
