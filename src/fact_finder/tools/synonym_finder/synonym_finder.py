from abc import ABC, abstractmethod
from typing import Iterable, List


class SynonymFinder(ABC):

    @abstractmethod
    def __call__(self, name: str) -> Iterable[str]:
        pass


class SimilaritySynonymFinder(SynonymFinder):

    def __call__(self, name: str) -> Iterable[str]:
        # TODO find most similar node from all nodes with vector cosine similarity search
        raise NotImplementedError


class SubWordSynonymFinder(SynonymFinder):

    def __call__(self, name: str) -> Iterable[str]:
        # TODO 2010.11784.pdf (arxiv.org)
        raise NotImplementedError
