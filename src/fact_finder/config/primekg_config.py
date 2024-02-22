from typing import List, Tuple

from fact_finder.prompt_templates import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from fact_finder.qa_service.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor
from fact_finder.qa_service.cypher_preprocessors.format_preprocessor import FormatPreprocessor
from fact_finder.qa_service.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)
from fact_finder.tools.sub_graph_extractor import LLMSubGraphExtractor
from fact_finder.qa_service.cypher_preprocessors.synonym_cypher_query_preprocessor import SynonymCypherQueryPreprocessor
from fact_finder.qa_service.neo4j_langchain_qa_service import Neo4JLangchainQAService
from fact_finder.tools.synonym_finder.preferred_term_finder import PreferredTermFinder
from fact_finder.tools.synonym_finder.wiki_data_synonym_finder import WikiDataSynonymFinder
from fact_finder.utils import build_neo4j_graph
from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.prompt import PromptTemplate

_USING_SYNONYMIZED_GRAPH = False

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
    preprocs: List[CypherQueryPreprocessor] = []
    preprocs.append(FormatPreprocessor())
    preprocs.append(LowerCasePropertiesCypherQueryPreprocessor())
    wikidata = WikiDataSynonymFinder()
    preprocs.append(SynonymCypherQueryPreprocessor(graph=graph, synonym_finder=wikidata, node_types="exposure"))
    if _USING_SYNONYMIZED_GRAPH:
        preprocs += _get_synonymized_graph_preprocessors(graph)
    return preprocs


def _get_synonymized_graph_preprocessors(graph: Neo4jGraph) -> List[CypherQueryPreprocessor]:
    preprocs: List[CypherQueryPreprocessor] = []
    gene_ent = PreferredTermFinder(["gene"])
    preprocs.append(SynonymCypherQueryPreprocessor(graph=graph, synonym_finder=gene_ent, node_types="gene_protein"))
    drug_ent = PreferredTermFinder(["drug"])
    preprocs.append(SynonymCypherQueryPreprocessor(graph=graph, synonym_finder=drug_ent, node_types="drug"))
    disease_ent = PreferredTermFinder(["disease"])
    preprocs.append(SynonymCypherQueryPreprocessor(graph=graph, synonym_finder=disease_ent, node_types="disease"))
    anatomy_ent = PreferredTermFinder(["Organs"])
    preprocs.append(SynonymCypherQueryPreprocessor(graph=graph, synonym_finder=anatomy_ent, node_types="anatomy"))
    return preprocs


def _get_graph_prompt_templates() -> Tuple[PromptTemplate, PromptTemplate]:
    return CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
