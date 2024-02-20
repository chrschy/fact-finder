import re
from typing import Optional, Set

from langchain_community.graphs import Neo4jGraph

from fact_finder.qa_service.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor
from fact_finder.tools.synonym_finder import SynonymFinder


class SynonymCypherQueryPreprocessor(CypherQueryPreprocessor):
    def __init__(self, graph: Neo4jGraph, synonym_finder: SynonymFinder):
        self.__graph = graph
        self.__cypher_query_to_get_all_nodes = "MATCH(n) RETURN n"
        self.__all_nodes = self.__get_all_nodes()
        self.__synonym_finder = synonym_finder

    def __get_all_nodes(self) -> Set[str]:
        nodes = self.__graph.query(self.__cypher_query_to_get_all_nodes)
        return set(node["n"]["name"].lower() for node in nodes)

    def __call__(self, cypher_query: str) -> str:
        node = self.__extract_node_from_cypher(cypher_query)
        if node is None:
            return cypher_query
        if not self.__is_node_in_graph(node):
            synonyms = self.__synonym_finder(node)
            for synonym in synonyms:
                if self.__is_node_in_graph(synonym):
                    return cypher_query.replace(node, synonym)
            return cypher_query
        return cypher_query

    def __is_node_in_graph(self, node: str) -> bool:
        return node.lower() in self.__all_nodes

    def __extract_node_from_cypher(self, cypher_query: str) -> Optional[str]:
        regex = r"{name: ['\"]([^}]+)['\"]}"
        matches = re.findall(regex, cypher_query, re.MULTILINE)
        if len(matches):
            return matches[0]
        else:
            return None
