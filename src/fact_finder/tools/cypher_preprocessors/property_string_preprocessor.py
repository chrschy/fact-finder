from abc import ABC, abstractmethod
from typing import List

import regex as re
from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import (
    CypherQueryPreprocessor,
)


class PropertyStringCypherQueryPreprocessor(CypherQueryPreprocessor, ABC):
    def __init__(self, property_names: List[str] = [r"[^{:\s]+"]) -> None:
        self._property_names = property_names

    def __call__(self, cypher_query: str) -> str:
        for name in self._property_names:
            cypher_query = self._replace_property_value_in_node(cypher_query, name)
            cypher_query = self._replace_property_value_in_comparison(cypher_query, name)
            cypher_query = self._replace_property_value_in_list(cypher_query, name)
        return cypher_query

    def _replace_property_value_in_node(self, cypher_query: str, name: str):
        return re.sub(r"\{" + name + r': "([^"}]+)"\}', self._replace_match, cypher_query)

    def _replace_property_value_in_comparison(self, cypher_query: str, name: str):
        return re.sub(r"[^\s=]+\." + name + r'\s*=\s*"([^"]+)"(\s|$)', self._replace_match, cypher_query)

    def _replace_property_value_in_list(self, cypher_query: str, name: str):
        return re.sub(r"\." + name + r'\s*IN\s*\[(?:(?:"([^"]+)"(?:,\s*)*)+?)\]', self._replace_match, cypher_query)

    @abstractmethod
    def _replace_match(self, matches: re.Match[str]) -> str: ...
