import argparse
from typing import List, Tuple

from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.prompt import PromptTemplate

from fact_finder.chains.graph_qa_chain import GraphQAChain
from fact_finder.prompt_templates import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor
from fact_finder.tools.cypher_preprocessors.format_preprocessor import FormatPreprocessor
from fact_finder.tools.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)
from fact_finder.tools.cypher_preprocessors.synonym_cypher_query_preprocessor import SynonymCypherQueryPreprocessor
from fact_finder.tools.synonym_finder.preferred_term_finder import PreferredTermFinder
from fact_finder.tools.synonym_finder.wiki_data_synonym_finder import WikiDataSynonymFinder
from fact_finder.utils import build_neo4j_graph


def build_chain(model: BaseLanguageModel, args: List[str] = []) -> Chain:
    parsed_args = _parse_primekg_args(args)
    graph = build_neo4j_graph()
    cypher_preprocessors = _build_preprocessors(graph, parsed_args.normalized_graph)
    cypher_prompt, answer_generation_prompt = _get_graph_prompt_templates()
    return GraphQAChain(
        llm=model,
        graph=graph,
        cypher_prompt=cypher_prompt,
        answer_generation_prompt=answer_generation_prompt,
        cypher_query_preprocessors=cypher_preprocessors,
        return_intermediate_steps=True,
    )


def _build_preprocessors(graph: Neo4jGraph, using_normalized_graph: bool) -> List[CypherQueryPreprocessor]:
    preprocs: List[CypherQueryPreprocessor] = []
    preprocs.append(FormatPreprocessor())
    preprocs.append(LowerCasePropertiesCypherQueryPreprocessor())
    wikidata = WikiDataSynonymFinder()
    preprocs.append(SynonymCypherQueryPreprocessor(graph=graph, synonym_finder=wikidata, node_types="exposure"))
    if using_normalized_graph:
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


def _parse_primekg_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized_graph", action="store_true")
    parsed_args, _ = parser.parse_known_args(args)
    return parsed_args
