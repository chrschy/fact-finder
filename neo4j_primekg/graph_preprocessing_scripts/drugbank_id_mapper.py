from typing import Optional, Set, Tuple

import pandas as pd
from id_based_mapper import IdBasedMapper
from mapper import _POLARS_AVAILABLE


class DrugbankIdMapper(IdBasedMapper):
    def __init__(
        self,
        graph: pd.DataFrame,
        drugbank_refs_file: str,
        drugbank_id_key: str = "DrugBank ID",
        new_id_key1: str = "SCI_DRUG ID",
        new_id_key2: str = "ID",
        name_key: str = "Preferred_Name",
        id_to_id_map_sheet: str = "concordance",
        id_to_pref_names_sheet: str = "csvdata",
        use_polars: bool = _POLARS_AVAILABLE,
    ) -> None:
        super().__init__(graph, use_polars=use_polars)
        self._id_to_id_map = pd.read_excel(drugbank_refs_file, sheet_name=id_to_id_map_sheet)
        self._id_to_pref_names = pd.read_excel(drugbank_refs_file, sheet_name=id_to_pref_names_sheet)
        self._drugbank_id_key = drugbank_id_key
        self._new_id_key1 = new_id_key1
        self._new_id_key2 = new_id_key2
        self._name_key = name_key

    def _get_relevant_node_types(self) -> Set[str]:
        return set(["drug"])

    def _get_mapping_for_id(self, graph_id: str) -> Tuple[str, Optional[str]]:
        id_to_id_res = self._id_to_id_map[self._id_to_id_map[self._drugbank_id_key] == graph_id]
        if len(id_to_id_res) == 0:
            return graph_id, None
        new_id = id_to_id_res[self._new_id_key1].iloc[0]
        new_names = self._id_to_pref_names[self._id_to_pref_names[self._new_id_key2] == new_id][self._name_key]
        if len(new_names) == 0:
            return graph_id, None
        new_name = new_names.iloc[0]
        return new_id, new_name
