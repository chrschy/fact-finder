import json
import os
import sys
from typing import Dict, Iterable, List, Set, Tuple, Union

from tqdm import tqdm

_POLARS_AVAILABLE = False
try:
    import polars as pd

    _POLARS_AVAILABLE = True
except:
    import logging

    import pandas as pd

    logging.warning("Polars not available for CSV processing. Using pandas which might be slow.")


def primekg_main(args: List[str]):
    if len(args) != 4:
        print(f"Usage: python {args[0]} <prime/kg.csv> <output/kg.csv> <mapping/directory/>")
        exit(1)

    graph_file = args[1]
    output_file = args[2]
    mapping_dir = args[3]

    print(f'Loading graph from "{graph_file}...')
    graph = pd.read_csv(graph_file, low_memory=False, infer_schema_length=0)
    print(f'Loading mappings from "{mapping_dir}...')
    mapper_fct = Mapper(mapping_dir, graph)
    print(f"Applying mappings to graph...")
    if _POLARS_AVAILABLE:
        col_names = graph.columns
        graph = graph.map_rows(mapper_fct)
        graph.columns = col_names
    else:
        tqdm.pandas()
        graph = graph.progress_apply(mapper_fct, axis="columns", result_type="broadcast")
    print(f'Writing transformed graph to "{output_file}"...')
    if _POLARS_AVAILABLE:
        graph.write_csv(output_file)
    else:
        graph.to_csv(output_file)


_allowed_categories_for_each_type: Dict[str, List[str]] = {
    "gene/protein": [
        "Gene",
        # "Disease",
        # "CellLine",
        # "Drug",
        # "Organs"
    ],
    "drug": [
        # "Gene",
        # "Antibody",
        # "Disease",
        # "CellLine",
        # "Cells",
        "Drug",
        # "Organs"
    ],
    "effect/phenotype": [
        # "Gene",
        # "Antibody",
        # "Disease",
        # "CellLine",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "disease": [
        # "Gene",
        # "Antibody",
        "Disease",
        # "CellLine",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "biological_process": [
        # "Gene",
        # "Antibody",
        # "Disease",
        # "CellLine",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "molecular_function": [
        # "Gene",
        # "Antibody",
        # "CellLine",
        # "Disease",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "cellular_component": [
        # "Gene",
        # "Antibody",
        # "Disease",
        # "CellLine",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "exposure": [
        # "Gene",
        # "Disease",
        # "CellLine",
        # "Drug",
        # "Organs"
    ],
    "pathway": [
        # "Gene",
        # "Antibody",
        # "CellLine",
        # "Disease",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "anatomy": [
        # "Gene",
        # "Disease",
        # "CellLine",
        # "Cells",
        # "Drug",
        "Organs"
    ],
}

allowed_categories_for_each_type: Dict[str, Set[str]] = {
    k: set(v) for k, v in _allowed_categories_for_each_type.items() if v
}


class Mapper:
    def __init__(self, directory: str, graph: pd.DataFrame) -> None:
        self._set_column_accessors(graph)
        self._mapping: Dict[Tuple[str, str], Tuple[str, str]] = dict(self._prepare_all_mappings(directory))

    def __call__(self, row: Union["pd.Series", Tuple[str, ...]]) -> Union["pd.Series", Tuple[str]]:
        if _POLARS_AVAILABLE:
            return self._call_polars(row)
        return self._call_pandas(row)

    def _call_polars(self, row: Tuple[str, ...]) -> Tuple[str]:
        row_lst = list(row)
        node_type, val_in_graph = row_lst[self._x_type_idx], row_lst[self._x_name_idx]
        if pref_name_and_id := self._mapping.get((node_type, val_in_graph)):
            row_lst[self._x_name_idx] = pref_name_and_id[0]
            row_lst[self._x_id_idx] = pref_name_and_id[1]
        node_type, val_in_graph = row_lst[self._y_type_idx], row_lst[self._y_name_idx]
        if pref_name_and_id := self._mapping.get((node_type, val_in_graph)):
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

    def _set_column_accessors(self, df: pd.DataFrame):
        if _POLARS_AVAILABLE:
            self._x_type_idx = df.get_column_index("x_type")
            self._x_name_idx = df.get_column_index("x_name")
            self._x_id_idx = df.get_column_index("x_id")
            self._y_type_idx = df.get_column_index("y_type")
            self._y_name_idx = df.get_column_index("y_name")
            self._y_id_idx = df.get_column_index("y_id")
        else:
            self._x_type_idx = "x_type"
            self._x_name_idx = "x_name"
            self._x_id_idx = "x_id"
            self._y_type_idx = "y_type"
            self._y_name_idx = "y_name"
            self._y_id_idx = "y_id"

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


if __name__ == "__main__":
    primekg_main(sys.argv)
