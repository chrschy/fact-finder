import re
from typing import List


class LowerCasePropertiesCypherQueryPreprocessor:
    def __init__(self, property_names: List[str] = [r"[^{:]+"]) -> None:
        self._property_names = property_names

    def __call__(self, cypher_query: str) -> str:
        for name in self._property_names:
            cypher_query = re.sub(
                r"{" + name + ": ['\"]([^'\"]+)['\"]}",
                lambda m: m.group(0).replace(m.group(1), m.group(1).lower()),
                cypher_query,
            )
        return cypher_query
