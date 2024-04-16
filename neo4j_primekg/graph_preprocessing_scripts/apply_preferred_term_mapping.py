import argparse
from typing import List

try:
    import polars as pd
except:
    import pandas as pd

from drugbank_id_mapper import DrugbankIdMapper
from id_based_mapper import IdBasedMapper
from mapper import _POLARS_AVAILABLE
from mondo_id_mapper import MondoIdMapper
from preferred_term_mapper import PreferredTermMapper


def primekg_main(args: argparse.Namespace):
    graph_file = args.graph_file
    output_file = args.output_file
    mapping_dir = args.mapping_dir
    mondo_id_mapping = args.mondo_id_mapping
    drugbank_id_mapping = args.drugbank_id_mapping

    print(f'Loading graph from "{graph_file}...')
    graph = pd.read_csv(graph_file, low_memory=False, infer_schema_length=0)
    id_based_mappers: List[IdBasedMapper] = []
    if mondo_id_mapping is not None:
        print(f'Loading Mondo id mapping from "{mondo_id_mapping}...')
        id_based_mappers.append(MondoIdMapper(graph, mondo_id_mapping))
    if drugbank_id_mapping is not None:
        print(f'Loading DrugBank id mapping from "{drugbank_id_mapping}...')
        id_based_mappers.append(DrugbankIdMapper(graph, drugbank_id_mapping))
    print(f"Applying id based mappings...")
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--mapping_dir", type=str, required=True)
    parser.add_argument("--mondo_id_mapping", type=str, default=None)
    parser.add_argument("--drugbank_id_mapping", type=str, default=None)

    primekg_main(parser.parse_args())
