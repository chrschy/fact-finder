from abc import ABC, abstractmethod


class Score(ABC):
    @abstractmethod
    def compare(self, text_a: str, text_b: str) -> float: ...
