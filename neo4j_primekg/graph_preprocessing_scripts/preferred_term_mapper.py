import json
import os
from typing import Dict, Iterable, List, Set, Tuple

from tqdm import tqdm

try:
    import polars as pd
except:
    import pandas as pd

from allowed_preferred_name_mapping_categories import allowed_categories_for_each_type
from mapper import Mapper


class PreferredTermMapper(Mapper):
    def __init__(self, directory: str, graph: pd.DataFrame) -> None:
        super().__init__(graph)
        self._mapping: Dict[Tuple[str, str], Tuple[str, str]] = dict(self._prepare_all_mappings(directory))
        self._filter_substr = "BAY"

    def _call_polars(self, row: Tuple[str, ...]) -> Tuple[str, ...]:
        row_lst = list(row)
        node_type, val_in_graph = row_lst[self._x_type_idx], row_lst[self._x_name_idx]
        if pref_name_and_id := self._mapping.get((node_type, val_in_graph)):
            if self._filter_substr not in pref_name_and_id[1]:
                row_lst[self._x_name_idx] = pref_name_and_id[0]
                row_lst[self._x_id_idx] = pref_name_and_id[1]
        node_type, val_in_graph = row_lst[self._y_type_idx], row_lst[self._y_name_idx]
        if pref_name_and_id := self._mapping.get((node_type, val_in_graph)):
            if self._filter_substr not in pref_name_and_id[1]:
                row_lst[self._y_name_idx] = pref_name_and_id[0]
                row_lst[self._y_id_idx] = pref_name_and_id[1]
        return tuple(row_lst)

    def _call_pandas(self, row: "pd.Series") -> "pd.Series":
        node_type, val_in_graph = row[[self._x_type_idx, self._x_name_idx]]
        if pref_name_and_id := self._mapping.get((node_type, val_in_graph)):
            row[[self._x_name_idx, self._x_id_idx]] = pref_name_and_id
        node_type, val_in_graph = row[[self._y_type_idx, self._y_name_idx]]
        if pref_name_and_id := self._mapping.get((node_type, val_in_graph)):
            row[[self._y_name_idx, self._y_id_idx]] = pref_name_and_id
        return row

    def _prepare_all_mappings(self, directory: str) -> Iterable[Tuple[Tuple[str, str], Tuple[str, str]]]:
        desc = "Loading mappings for categories"
        for node_type, allowed_categories in tqdm(allowed_categories_for_each_type.items(), desc=desc):
            mappings = self._load_mappings(directory, node_type)
            yield from self._create_allowed_mapping(node_type, allowed_categories, mappings)

    def _load_mappings(self, directory: str, node_type: str) -> Dict[str, List[Tuple[str, str, str]]]:
        fn = os.path.join(directory, f"{node_type}_data.json".replace("/", "_").replace(" ", "_"))
        with open(fn, "r") as f:
            return json.load(f)

    def _create_allowed_mapping(
        self,
        node_type: str,
        ordered_allowed_categories: Set[str],
        mappings: Dict[str, List[Tuple[str, str, str]]],
    ) -> Iterable[Tuple[Tuple[str, str], Tuple[str, str]]]:
        for val_in_graph, pref_names in mappings.items():
            pref_names = list(filter(lambda pn: pn[2] in ordered_allowed_categories, pref_names))
            if len(pref_names) == 0:
                continue
            entry_per_category = {category: (pref_name, id) for pref_name, id, category in pref_names}
            if len(entry_per_category) != len(pref_names):
                continue  #  Two synonyms in the same category are considered an error.
            most_relevant_category = next(
                category for category in ordered_allowed_categories if category in entry_per_category
            )
            pref_name, id = entry_per_category[most_relevant_category]
            yield (node_type, val_in_graph), (pref_name, id)
