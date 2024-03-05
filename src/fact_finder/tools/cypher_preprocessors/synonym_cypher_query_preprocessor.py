import re
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from langchain_community.graphs import Neo4jGraph

from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor
from fact_finder.tools.synonym_finder.synonym_finder import SynonymFinder


class SynonymCypherQueryPreprocessor(CypherQueryPreprocessor):

    def __init__(
        self,
        graph: Neo4jGraph,
        synonym_finder: SynonymFinder,
        node_types: str | List[str],
        search_property_name: str = "name",
        replacement_property_name: str | None = None,
    ):
        self.__graph = graph
        self.__synonym_finder = synonym_finder
        self.__node_types = {node_types} if isinstance(node_types, str) else set(node_types)
        self.__search_property_name = search_property_name
        self.__replacement_property_name = (
            replacement_property_name if replacement_property_name else self.__search_property_name
        )
        self.__existing_node_properties = dict(self.__get_all_nodes())

    def __call__(self, cypher_query: str) -> str:
        for node_type in self.__node_types:
            regex = self._build_match_clause_regex(node_type)
            cypher_query = re.sub(regex, partial(self._replace_match, node_type=node_type), cypher_query)
        for node_type in self.__node_types:
            regex = self._build_where_clause_regex(node_type)
            cypher_query = re.sub(
                regex,
                partial(self._replace_match, node_type=node_type, group_offset=1),
                cypher_query,
                flags=re.MULTILINE | re.DOTALL,
            )
        return cypher_query

    def __get_all_nodes(self) -> Iterable[Tuple[str, Set[str]]]:
        for n_type in self.__node_types:
            cypher_query = f"MATCH(n:{n_type}) RETURN n"
            nodes = self.__graph.query(cypher_query)
            yield n_type, set(node["n"][self.__replacement_property_name].lower() for node in nodes)

    def _build_match_clause_regex(self, node_type) -> str:
        return r"\([^\s{:]+:" + node_type + r"\s*{(" + self.__search_property_name + r"): ['\"]([^}]+)['\"]}\)"

    def _build_where_clause_regex(self, node_type) -> str:
        return (
            r"MATCH[^$]*\(([^\s{:]+):" + node_type + r"\).*?"
            r"WHERE[^$]*?\1\.(" + self.__search_property_name + r")\s*=\s*\"([^\"]+)\""
        )

    def _replace_match(self, match: re.Match[str], node_type: str, group_offset: int = 0) -> str:
        assert len(match.groups()) == 2 + group_offset
        text = match.group(0)
        extracted_property = match.group(2 + group_offset)
        replacement_property = self.__find_synonym(node_type, extracted_property)
        if replacement_property is None:
            return text
        return self._build_new_text(text, match, group_offset, replacement_property)

    def _build_new_text(self, text: str, match: re.Match[str], group_offset: int, replacement_property: str) -> str:
        hit_offset = match.start(0)
        start1 = match.start(1 + group_offset) - hit_offset
        end1 = match.end(1 + group_offset) - hit_offset
        start2 = match.start(2 + group_offset) - hit_offset
        end2 = match.end(2 + group_offset) - hit_offset
        new_text = (
            text[:start1] + self.__replacement_property_name + text[end1:start2] + replacement_property + text[end2:]
        )
        return new_text

    def __find_synonym(self, node_type: str, node_property: str) -> str | None:
        synonyms = self.__synonym_finder(node_property)
        for synonym in synonyms:
            if self.__exists_in_graph(node_type, synonym):
                return synonym
        return None

    def __exists_in_graph(self, node_type: str, property_value: str) -> bool:
        return property_value.lower() in self.__existing_node_properties[node_type]
