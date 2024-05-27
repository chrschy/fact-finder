import re

from fact_finder.tools.cypher_preprocessors.cypher_query_preprocessor import (
    CypherQueryPreprocessor,
)


class AlwaysDistinctCypherQueryPreprocessor(CypherQueryPreprocessor):
    def __call__(self, cypher_query: str) -> str:
        return re.sub(r"RETURN\s+(?!DISTINCT).*", self._replace_match, cypher_query)

    def _replace_match(self, matches: re.Match[str]) -> str:
        return matches.group(0).replace("RETURN", "RETURN DISTINCT")
