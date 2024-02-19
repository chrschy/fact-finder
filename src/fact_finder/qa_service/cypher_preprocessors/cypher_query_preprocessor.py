from typing import Protocol, runtime_checkable


@runtime_checkable
class CypherQueryPreprocessor(Protocol):
    def __call__(self, cypher_query: str) -> str: ...
