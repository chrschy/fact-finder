from typing import List, Tuple

from fact_finder.prompt_templates import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from fact_finder.qa_service.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor
from fact_finder.qa_service.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)
from fact_finder.qa_service.cypher_preprocessors.synonym_cypher_query_preprocessor import SynonymCypherQueryPreprocessor
from fact_finder.qa_service.neo4j_langchain_qa_service import Neo4JLangchainQAService
from fact_finder.synonym_finder.synonym_finder import WikiDataSynonymFinder
from fact_finder.utils import build_neo4j_graph
from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.prompt import PromptTemplate


def build_chain(model: BaseLanguageModel) -> Chain:
    graph = build_neo4j_graph()
    cypher_preprocessors = _build_preprocessors(graph)
    cypher_prompt, qa_prompt = _get_graph_prompt_templates()
    return Neo4JLangchainQAService.from_llm(
        model,
        graph=graph,
        cypher_query_preprocessors=cypher_preprocessors,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        verbose=True,
        return_intermediate_steps=True,
    )


def _build_preprocessors(graph: Neo4jGraph) -> List[CypherQueryPreprocessor]:
    lower_case_preprocessor = LowerCasePropertiesCypherQueryPreprocessor()
    synonym_preprocessor = SynonymCypherQueryPreprocessor(graph=graph, synonym_finder=WikiDataSynonymFinder())
    return [lower_case_preprocessor, synonym_preprocessor]


def _get_graph_prompt_templates() -> Tuple[PromptTemplate, PromptTemplate]:
    return CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
