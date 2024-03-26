from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple, Union

import pandas as pd
from tqdm import tqdm


def primekg_main(args: List[str]):
    from fact_finder.tools.entity_detector import EntityDetector

    if len(args) != 2:
        print(f"Usage: python {args[0]} <prime/kg.csv>")
        exit(1)

    graph = pd.read_csv(args[1], low_memory=False)
    detector = EntityDetector()
    extractor = PreferredTermDataExtractor(graph=graph, api_tool=detector)

    graph_types = [
        "gene/protein",
        "drug",
        "effect/phenotype",
        "disease",
        "biological_process",
        "molecular_function",
        "cellular_component",
        "exposure",
        "pathway",
        "anatomy",
    ]

    for graph_type in graph_types:
        print("---" * 20)
        print(f"PROCESSING {graph_type}")
        extractor.process(graph_type)
        print("---" * 20)


class PreferredTermMapping(dict):
    def __init__(self, data: Iterable[Tuple[str, List[Tuple[str, str, str]]]]):
        super().__init__(data)

    @classmethod
    def from_dataframe(
        cls,
        graph_data: pd.DataFrame,
        graph_type: str,
        api_tool: Callable[[str], List[Dict[str, Any]]],
        type_key: str = "x_type",
        name_key: str = "x_name",
    ) -> PreferredTermMapping:
        return cls(
            cls._map_entity_to_prefered_terms_and_categories(graph_data, graph_type, api_tool, type_key, name_key)
        )

    @staticmethod
    def _map_entity_to_prefered_terms_and_categories(
        graph_data: pd.DataFrame,
        graph_type: str,
        api_tool: Callable[[str], List[Dict[str, Any]]],
        type_key: str = "x_type",
        name_key: str = "x_name",
    ) -> Iterable[Tuple[str, List[Tuple[str, str, str]]]]:
        relevant_entries = graph_data[graph_data[type_key] == graph_type][name_key].unique()
        for entry in tqdm(relevant_entries):
            pref_term_results = PreferredTermMapping._extract_preferred_name_and_id_and_category(entry, api_tool)
            yield entry, pref_term_results

    @staticmethod
    def _extract_preferred_name_and_id_and_category(
        name: str, api_tool: Callable[[str], List[Dict[str, Any]]]
    ) -> List[Tuple[str, str, str]]:
        return [(r["pref_term"], r["id"], r["sem_type"]) for r in api_tool(name)]


class PreferredTermDataExtractor:
    def __init__(
        self,
        graph: Union[str, pd.DataFrame],
        api_tool: Callable[[str], List[Dict[str, Any]]],
        type_key: str = "x_type",
        name_key: str = "x_name",
    ) -> None:
        self._api_tool = api_tool
        self._graph = pd.read_csv(graph, low_memory=False) if isinstance(graph, str) else graph
        self._type_key = type_key
        self._name_key = name_key

    def process(self, graph_type: str):
        print(f">> Parsing preferred terms for category {graph_type}...")
        data = PreferredTermMapping.from_dataframe(
            graph_data=self._graph,
            graph_type=graph_type,
            api_tool=self._api_tool,
            type_key=self._type_key,
            name_key=self._name_key,
        )
        print("Data:", json.dumps(data))
        self.store(data, graph_type)
        self.collect_categories(data)
        self.print_errors(data)

    @staticmethod
    def store(data: PreferredTermMapping, graph_type: str):
        fn = f"{graph_type}_data.json".replace("/", "_").replace(" ", "_")
        print(f">> Storing results as {fn}...")
        with open(fn, "w") as file:
            json.dump(data, file)

    @staticmethod
    def collect_categories(data: PreferredTermMapping) -> Set[str]:
        print(f">> Extracting unique categories...")
        categories = set()
        for v in data.values():
            for entries in v:
                categories.add(entries[2])
        print(f">> Categories found: {list(categories)}")
        return categories

    @staticmethod
    def print_errors(data: PreferredTermMapping) -> None:
        print(">> The following entries have duplicate category errors:")
        for k, v in data.items():
            if len(v) > 1:
                unique_categories = set(entries[2] for entries in v)
                if len(v) != len(unique_categories):
                    print(f">>>> Error for drug '{k}': Entries are {v}")


if __name__ == "__main__":
    primekg_main(sys.argv)
