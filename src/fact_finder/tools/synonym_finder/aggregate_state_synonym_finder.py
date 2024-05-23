import logging
from typing import List

from fact_finder.tools.synonym_finder.synonym_finder import SynonymFinder


class AggregateStateSynonymFinder(SynonymFinder):

    def __call__(self, name: str) -> List[str]:
        name = name.strip().lower()
        if name in ["gas", "gases", "gaseous", "gassy", "gasiform", "aerially", "aeriform", "vapor", "vapour"]:
            return ["gas"]
        if name in ["liquid", "liquids", "fluid", "fluids"]:
            return ["liquid"]
        if name in ["solid", "solids"]:
            return ["solid"]
        logging.error(f"Unknown aggregate state: {name}")
        return [name]
