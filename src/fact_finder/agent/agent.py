from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def answer(self, question: str) -> str:
        ...
