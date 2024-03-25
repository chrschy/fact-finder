import argparse
from typing import Dict, List, Tuple

from fact_finder.chains.graph_qa_chain.config import GraphQAChainConfig
from fact_finder.chains.graph_qa_chain.graph_qa_chain import GraphQAChain
from fact_finder.chains.graph_summary_chain import GraphSummaryChain
from fact_finder.config.primekg_predicate_descriptions import PREDICATE_DESCRIPTIONS
from fact_finder.prompt_templates import (
    COMBINED_QA_PROMPT,
    CYPHER_GENERATION_PROMPT,
    CYPHER_QA_PROMPT,
    KEYWORD_PROMPT,
    SUBGRAPH_SUMMARY_PROMPT,
)
from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import (
    CypherQueryPreprocessor,
)
from fact_finder.tools.cypher_preprocessors.format_preprocessor import (
    FormatPreprocessor,
)
from fact_finder.tools.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)
from fact_finder.tools.cypher_preprocessors.synonym_cypher_query_preprocessor import (
    SynonymCypherQueryPreprocessor,
)
from fact_finder.tools.entity_detector import EntityDetector
from fact_finder.tools.synonym_finder.preferred_term_finder import PreferredTermFinder
from fact_finder.tools.synonym_finder.wiki_data_synonym_finder import (
    WikiDataSynonymFinder,
)
from fact_finder.utils import build_neo4j_graph
from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.prompt import PromptTemplate


def build_chain(model: BaseLanguageModel, args: List[str] = []) -> Chain:
    parsed_args = _parse_primekg_args(args)
    graph = build_neo4j_graph()
    cypher_preprocessors = _build_preprocessors(graph, parsed_args.normalized_graph)
    cypher_prompt, answer_generation_prompt = _get_graph_prompt_templates()
    config = GraphQAChainConfig(
        llm=model,
        graph=graph,
        cypher_prompt=cypher_prompt,
        answer_generation_prompt=answer_generation_prompt,
        cypher_query_preprocessors=cypher_preprocessors,
        predicate_descriptions=PREDICATE_DESCRIPTIONS[:10],
        return_intermediate_steps=True,
        use_entity_detection_preprocessing=parsed_args.use_entity_detection_preprocessing,
        entity_detector=EntityDetector() if parsed_args.use_entity_detection_preprocessing else None,
        allowed_types_and_description_templates=_get_primekg_entity_categories(),
        use_subgraph_expansion=parsed_args.use_subgraph_expansion,
        combine_output_with_sematic_scholar=parsed_args.combine_output_with_sematic_scholar,
        semantic_scholar_keyword_prompt=KEYWORD_PROMPT,
        combined_answer_generation_prompt=COMBINED_QA_PROMPT,
    )
    return GraphQAChain(config)


def build_chain_summary(model: BaseLanguageModel, args: List[str] = []) -> Chain:
    return GraphSummaryChain(llm=model, graph_summary_template=SUBGRAPH_SUMMARY_PROMPT, return_intermediate_steps=True)


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
    parser.add_argument("--use_subgraph_expansion", action="store_true")
    parser.add_argument("--combine_output_with_sematic_scholar", action="store_true")
    parser.add_argument("--use_entity_detection_preprocessing", action="store_true")
    parsed_args, _ = parser.parse_known_args(args)
    return parsed_args


def _get_primekg_entity_categories() -> Dict[str, str]:
    return {
        "disease": "{entity} is a disease.",
        "drug": "{entity} is a drug.",
        "gene": "{entity} is a gene.",
        "organs": "{entity} is a organ.",
    }
