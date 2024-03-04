from typing import Dict, Any

from dotenv import load_dotenv

from fact_finder.chains.cypher_preprocessors.format_preprocessor import FormatPreprocessor
from fact_finder.chains.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)
from fact_finder.chains.cypher_query_generation_chain import CypherQueryGenerationChain
from fact_finder.chains.cypher_query_preprocessors_chain import CypherQueryPreprocessorsChain
from fact_finder.chains.graph_chain import GraphChain
from fact_finder.chains.qa_chain import QAChain
from fact_finder.chains.subgraph_extractor_chain import SubgraphExtractorChain
from fact_finder.prompt_templates import CYPHER_GENERATION_PROMPT
from fact_finder.utils import load_chat_model, build_neo4j_graph


class CustomChain:

    def __init__(self, llm=load_chat_model(), graph=build_neo4j_graph()):
        cypher_query_generation_chain = CypherQueryGenerationChain(
            llm=llm, graph=graph, cypher_prompt=CYPHER_GENERATION_PROMPT
        )
        cypher_query_preprocessors = [FormatPreprocessor(), LowerCasePropertiesCypherQueryPreprocessor()]
        cypher_query_preprocessors_chain = CypherQueryPreprocessorsChain(
            cypher_query_preprocessors=cypher_query_preprocessors
        )
        graph_chain = GraphChain(graph=graph)
        qa_chain = QAChain(llm=llm)
        subgraph_extractor_chain = SubgraphExtractorChain(llm=llm, graph=graph)
        self.combined_chain = (
            cypher_query_generation_chain
            | cypher_query_preprocessors_chain
            | {
                subgraph_extractor_chain.output_key: subgraph_extractor_chain,
                qa_chain.output_key: graph_chain | qa_chain,
            }
        )

    def __call__(self, question: str) -> Dict[str, Any]:
        return self.combined_chain.invoke(question)


if __name__ == "__main__":
    load_dotenv()
    custom_chain = CustomChain()
    questions = [
        "Which drugs are associated with epilepsy?",
        "Which drugs are associated with schizophrenia?",
        "Which medication has the most indications?",
        "What are the phenotypes associated with cardioacrofacial dysplasia?",
    ]
    for question in questions:
        result = custom_chain(question)
