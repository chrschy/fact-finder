import re

from langchain_community.graphs import Neo4jGraph

from fact_finder.qa_service.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor
from fact_finder.synonym_finder.synonym_finder import SynonymFinder


class SynonymCypherQueryPreprocessor(CypherQueryPreprocessor):
    def __init__(self, graph: Neo4jGraph, synonym_finder: SynonymFinder):
        self.__graph = graph
        self.__cypher_query_to_get_all_nodes = "MATCH(n) RETURN n"
        self.__all_nodes = self.__get_all_nodes()
        self.__synonym_finder = synonym_finder

    def __get_all_nodes(self) -> set:
        nodes = self.__graph.query(self.__cypher_query_to_get_all_nodes)
        nodes = set([node["n"]["name"].lower() for node in nodes])
        return nodes

    def __call__(self, cypher_query: str) -> str:
        node = self.__extract_node_from_cypher(cypher_query)
        if not self.__is_node_in_graph(node):
            synonyms = self.__get_synonyms(node)
            for synonym in synonyms:
                if self.__is_node_in_graph(synonym):
                    return cypher_query.replace(node, synonym)
            return cypher_query
        return cypher_query

    def __is_node_in_graph(self, node: str) -> bool:
        return node.lower() in self.__all_nodes

    def __get_synonyms(self, node: str) -> list[str]:
        return self.__synonym_finder.find(node)

    def __extract_node_from_cypher(self, cypher_query: str) -> str:
        regex = r"{name: ['\"]([^'\"]+)['\"]}"
        matches = re.findall(regex, cypher_query, re.MULTILINE)
        return matches[0]
