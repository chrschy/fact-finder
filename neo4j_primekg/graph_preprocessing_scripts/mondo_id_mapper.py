from enum import StrEnum
from typing import List, Optional, Set, Tuple, Union

import pandas as pd
from id_based_mapper import IdBasedMapper


class MondoColumn(StrEnum):
    DIRECT_MATCHING = "MONDO-ID\nMATCHING ID(s)"
    CROSSREFERENCES_MATCHING = "MONDO-ID\nVIA CROSSREFS AND PREFERRED LABEL MATCH"
    LEVENSHTEIN_MATCHING = "cell2Vlookup_left_all[MONDO: AU→B]_A\nSYNONYMS MATCH (>75%Levenshtein)"


class MondoIdMapper(IdBasedMapper):
    def __init__(
        self,
        graph: pd.DataFrame,
        mondo_refs_file: str,
        mondo_id_key: Union[MondoColumn, str] = MondoColumn.LEVENSHTEIN_MATCHING,
        new_id_key: str = "BAYER ID",
        name_key: str = "name",
        id_to_id_map_sheet: str = "concordance",
        id_to_pref_names_sheet: str = "SCI_DISEASE",
    ) -> None:
        super().__init__(graph)
        self._id_to_id_map = pd.read_excel(mondo_refs_file, sheet_name=id_to_id_map_sheet)
        self._id_to_pref_names = pd.read_excel(mondo_refs_file, sheet_name=id_to_pref_names_sheet)
        self._mondo_id_key = str(mondo_id_key)
        self._new_id_key = new_id_key
        self._name_key = name_key

    def _get_relevant_node_types(self) -> Set[str]:
        return set(["disease"])

    def _get_mapping_for_id(self, graph_id: str) -> Tuple[str, Optional[str]]:
        graph_id = f"MONDO:{int(graph_id):07d}"
        id_to_id_res = self._id_to_id_map[self._id_to_id_map[self._mondo_id_key] == graph_id]
        if len(id_to_id_res) == 0:
            return graph_id, None
        new_id = id_to_id_res[self._new_id_key].iloc[0]
        new_name = self._id_to_pref_names[self._id_to_pref_names[self._new_id_key] == new_id][self._name_key].iloc[0]
        return new_id, new_name
