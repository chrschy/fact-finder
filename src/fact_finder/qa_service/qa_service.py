from abc import ABC, abstractmethod


class QAService(ABC):

    @abstractmethod
    def search(self, user_query: str) -> str:
        ...
