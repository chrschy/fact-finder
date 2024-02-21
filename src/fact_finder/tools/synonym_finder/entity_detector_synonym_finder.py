from typing import Iterable, List

from fact_finder.tools.entity_detector import EntityDetector
from fact_finder.tools.synonym_finder.synonym_finder import SynonymFinder


class EntityDetectorSynonymFinder(SynonymFinder):
    """Searches and returns preferred names using the EntityDetector.
    Returns only prefered names of entities from the allow categories (semantic types).
    """

    def __init__(self, allowed_categories: Iterable[str]) -> None:
        self._detector = EntityDetector()
        self._allowed_categories = set(map(str.lower, allowed_categories))

    def __call__(self, name: str) -> List[str]:
        return [r["pref_term"] for r in self._detector(name) if r["sem_type"].lower() in self._allowed_categories]


class IdEntityDetectorSynonymFinder(SynonymFinder):
    """Searches preferred names using the EntityDetector.
    Returns the corresponding ids.
    Returns only ids of entities from the allow categories (semantic types).
    """

    def __init__(self, allowed_categories: Iterable[str]) -> None:
        self._detector = EntityDetector()
        self._allowed_categories = set(map(str.lower, allowed_categories))

    def __call__(self, name: str) -> List[str]:
        return [r["id"] for r in self._detector(name) if r["sem_type"].lower() in self._allowed_categories]
