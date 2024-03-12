import re
from typing import List

from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import CypherQueryPreprocessor


class LowerCasePropertiesCypherQueryPreprocessor(CypherQueryPreprocessor):
    def __init__(self, property_names: List[str] = [r"[^{:\s]+"]) -> None:
        self._property_names = property_names

    def __call__(self, cypher_query: str) -> str:
        for name in self._property_names:
            cypher_query = re.sub(
                r"{" + name + r': "([^}]+)"}',
                _replace_match_with_lower_case,
                cypher_query,
            )
            cypher_query = re.sub(
                r"[^\s=]+\." + name + r'\s*=\s*"([^"]+)"(\s|$)',
                _replace_match_with_lower_case,
                cypher_query,
            )
        return cypher_query


def _replace_match_with_lower_case(matches: re.Match[str]) -> str:
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), matches.group(1).lower())
