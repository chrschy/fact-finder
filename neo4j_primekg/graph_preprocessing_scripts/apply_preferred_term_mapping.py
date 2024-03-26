import sys
from typing import List

try:
    import polars as pd
except:
    import pandas as pd

from id_based_mapper import IdBasedMapper
from mapper import _POLARS_AVAILABLE
from mondo_id_mapper import MondoIdMapper
from preferred_term_mapper import PreferredTermMapper


def primekg_main(args: List[str]):
    if len(args) not in [4, 5]:
        print(f"Usage: python {args[0]} <prime/kg.csv> <output/kg.csv> <mapping/directory/> [mondo/id/mapping.xlsx]")
        exit(1)

    graph_file = args[1]
    output_file = args[2]
    mapping_dir = args[3]
    mondo_id_mapping = args[4] if len(args) == 5 else None

    print(f'Loading graph from "{graph_file}...')
    graph = pd.read_csv(graph_file, low_memory=False, infer_schema_length=0)
    if mondo_id_mapping is not None:
        print(f'Applying id based mappings "{graph_file}...')
        id_based_mappers: List[IdBasedMapper] = [MondoIdMapper(graph, mondo_id_mapping)]
        for mapping in id_based_mappers:
            graph = mapping.apply_to_graph(graph)
    print(f'Loading mappings from "{mapping_dir}...')
    mapper_fct = PreferredTermMapper(mapping_dir, graph)
    print(f"Applying mappings to graph...")
    graph = mapper_fct.apply_to_graph(graph)
    print(f'Writing transformed graph to "{output_file}"...')
    if _POLARS_AVAILABLE:
        graph.write_csv(output_file)
    else:
        graph.to_csv(output_file)


if __name__ == "__main__":
    primekg_main(sys.argv)
