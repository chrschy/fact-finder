import re
import sys

from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import (
    CypherQueryPreprocessor,
)


class FormatPreprocessor(CypherQueryPreprocessor):
    """A preprocessor applying some regex based formating to cypher queries.
    This allows simpler regex expressions in subsequent preprocessors.
    Partially based on https://github.com/TristanPerry/cypher-query-formatter
    (see MIT License: https://github.com/TristanPerry/cypher-query-formatter/blob/master/LICENSE).

    :param CypherQueryPreprocessor: _description_
    :type CypherQueryPreprocessor: _type_
    """

    def __call__(self, cypher_query: str) -> str:
        try:
            return self._try_formatting(cypher_query)
        except Exception as e:
            print(f"Cypher query formatting failed: {e}", file=sys.stderr)
            return cypher_query

    def _try_formatting(self, cypher_query: str) -> str:
        cypher_query = self._only_use_double_quotes(cypher_query)
        cypher_query = self._keywords_to_upper_case(cypher_query)
        cypher_query = self._null_and_boolean_literals_to_lower_case(cypher_query)
        cypher_query = self._ensure_main_keywords_on_newline(cypher_query)
        cypher_query = self._unix_style_newlines(cypher_query)
        cypher_query = self._remove_whitespace_from_start_of_lines(cypher_query)
        cypher_query = self._remove_whitespace_from_end_of_lines(cypher_query)
        cypher_query = self._add_spaces_after_comma(cypher_query)
        cypher_query = self._multiple_spaces_to_single_space(cypher_query)
        cypher_query = self._indent_on_create_and_on_match(cypher_query)
        cypher_query = self._remove_multiple_empty_newlines(cypher_query)
        cypher_query = self._remove_unnecessary_spaces(cypher_query)

        return cypher_query.strip()

    def _only_use_double_quotes(self, cypher_query: str) -> str:
        # Escape all single quotes in double quote strings.
        cypher_query = re.sub(
            r'"([^{}\(\)\[\]=]*?)"', lambda matches: matches.group(0).replace("'", r"\'"), cypher_query
        )
        # Replace all not escaped single quotes.
        cypher_query = re.sub(r"(?<!\\)'", lambda m: m.group(0)[:-1] + '"', cypher_query)
        return cypher_query

    def _keywords_to_upper_case(self, cypher_query: str) -> str:
        return re.sub(
            r"\b(WHEN|CASE|AND|OR|XOR|DISTINCT|AS|IN|STARTS WITH|ENDS WITH|CONTAINS|NOT|SET|ORDER BY)\b",
            _keywords_to_upper_case,
            cypher_query,
            flags=re.IGNORECASE,
        )

    def _null_and_boolean_literals_to_lower_case(self, cypher_query: str) -> str:
        return re.sub(
            r"\b(NULL|TRUE|FALSE)\b",
            _null_and_booleans_to_lower_case,
            cypher_query,
            flags=re.IGNORECASE,
        )

    def _ensure_main_keywords_on_newline(self, cypher_query: str) -> str:
        return re.sub(
            r"\b(CASE|DETACH DELETE|DELETE|MATCH|MERGE|LIMIT|OPTIONAL MATCH|RETURN|UNWIND|UNION|WHERE|WITH|GROUP BY)\b",
            _main_keywords_on_newline,
            cypher_query,
            flags=re.IGNORECASE,
        )

    def _unix_style_newlines(self, cypher_query: str) -> str:
        return re.sub(r"(\r\n|\r)", "\n", cypher_query)

    def _remove_whitespace_from_start_of_lines(self, cypher_query: str) -> str:
        return re.sub(r"^\s+", "", cypher_query, flags=re.MULTILINE)

    def _remove_whitespace_from_end_of_lines(self, cypher_query: str) -> str:
        return re.sub(r"\s+$", "", cypher_query, flags=re.MULTILINE)

    def _add_spaces_after_comma(self, cypher_query: str) -> str:
        return re.sub(r",([^\s])", lambda matches: ", " + matches.group(1), cypher_query)

    def _multiple_spaces_to_single_space(self, cypher_query: str) -> str:
        return re.sub(r"((?![\n])\s)+", " ", cypher_query)

    def _indent_on_create_and_on_match(self, cypher_query: str) -> str:
        return re.sub(r"\b(ON CREATE|ON MATCH)\b", _indent_on_create_and_on_match, cypher_query, flags=re.IGNORECASE)

    def _remove_multiple_empty_newlines(self, cypher_query: str) -> str:
        return re.sub(r"\n\s*?\n", "\n", cypher_query)

    def _remove_unnecessary_spaces(self, cypher_query: str) -> str:
        cypher_query = re.sub(r"(\(|{|\[])\s+", lambda matches: matches.group(1), cypher_query)
        cypher_query = re.sub(r"\s+(\)|}|\])", lambda matches: matches.group(1), cypher_query)
        cypher_query = re.sub(r"\s*(:|-|>|<)\s*", lambda matches: matches.group(1), cypher_query)
        # Retain spaces before property names
        cypher_query = re.sub(r':\s*"', ': "', cypher_query)
        # Also around equation signs
        cypher_query = re.sub(r"\s+=\s*", " = ", cypher_query)
        cypher_query = re.sub(r"\s*<=\s*", " <= ", cypher_query)
        cypher_query = re.sub(r"\s*>=\s*", " >= ", cypher_query)
        return cypher_query


def _keywords_to_upper_case(matches: re.Match[str]) -> str:
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), " " + matches.group(1).upper().strip() + " ")


def _null_and_booleans_to_lower_case(matches: re.Match[str]) -> str:
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), " " + matches.group(1).lower().strip() + " ")


def _main_keywords_on_newline(matches: re.Match[str]) -> str:
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), "\n" + matches.group(1).upper().lstrip() + " ")


def _indent_on_create_and_on_match(matches: re.Match[str]) -> str:
    assert len(matches.groups())
    return matches.group(0).replace(matches.group(1), "\n  " + matches.group(1).upper().lstrip() + " ")
