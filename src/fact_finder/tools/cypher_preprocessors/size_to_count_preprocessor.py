import re

from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import (
    CypherQueryPreprocessor,
)


class SizeToCountPreprocessor(CypherQueryPreprocessor):
    def __call__(self, cypher_query: str) -> str:
        cypher_query = re.sub(r"\b(SIZE\b\s*)\(", _size_to_upper_case, cypher_query, flags=re.IGNORECASE)
        cypher_query = self._replace_search_word_with_count(cypher_query, "SIZE(")
        return cypher_query

    def _replace_search_word_with_count(self, cypher_query, search_term):
        len_term = len(search_term)
        while (start := cypher_query.find(search_term)) >= 0:
            end = self._find_closing_bracket(cypher_query, start + len_term)
            cypher_query = (
                cypher_query[:start] + "COUNT{" + cypher_query[start + len_term : end] + "}" + cypher_query[end + 1 :]
            )
        return cypher_query

    def _find_closing_bracket(self, cypher_query: str, search_start: int, bracket_values={"(": 1, ")": -1}) -> int:
        count = 1
        for i, c in enumerate(cypher_query[search_start:]):
            count += bracket_values.get(c, 0)
            if count == 0:
                return i + search_start
        raise ValueError("SIZE keyword in Cypher query without closing bracket.")


def _size_to_upper_case(matches: re.Match[str]) -> str:
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), matches.group(1).upper().strip())
