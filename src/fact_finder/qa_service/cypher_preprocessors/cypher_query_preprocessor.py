from abc import ABC, abstractmethod


class CypherQueryPreprocessor(ABC):
    @abstractmethod
    def __call__(self, cypher_query: str) -> str:
        pass
