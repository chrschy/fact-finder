from typing import Iterable, List

from fact_finder.tools.entity_detector import EntityDetector
from fact_finder.tools.synonym_finder.synonym_finder import SynonymFinder


class PreferredTermFinder(SynonymFinder):
    """Searches and returns preferred names using the EntityDetector.
    Returns only preferred names of entities from the allowed categories (semantic types).
    """

    def __init__(self, allowed_categories: Iterable[str]) -> None:
        self._detector = EntityDetector()
        self._allowed_categories = set(map(str.lower, allowed_categories))

    def __call__(self, name: str) -> Iterable[str]:
        yield name
        for r in self._detector(name):
            if r["sem_type"].lower() in self._allowed_categories:
                yield r["pref_term"]


class PreferredTermIdFinder(SynonymFinder):
    """Searches preferred names using the EntityDetector.
    Returns the corresponding ids.
    Returns only ids of entities from the allowed categories (semantic types).
    """

    def __init__(self, allowed_categories: Iterable[str]) -> None:
        self._detector = EntityDetector()
        self._allowed_categories = set(map(str.lower, allowed_categories))

    def __call__(self, name: str) -> List[str]:
        return [r["id"] for r in self._detector(name) if r["sem_type"].lower() in self._allowed_categories]
