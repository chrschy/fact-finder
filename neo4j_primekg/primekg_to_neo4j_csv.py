import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


def extract_node_and_relationship_csvs(primekg_csv: str, target_dir: str, use_display_rel: bool) -> str:
    # fmt: off
    primekg_data =  load_csv_file(primekg_csv)
    primekg_data =  clean_data(primekg_data)
    node_dfs     =  build_nodes_by_type_dfs(primekg_data)
    rel_dfs      =  build_relationship_dfs(primekg_data, use_display_rel)
    node_csvs    =  create_node_csvs(target_dir, node_dfs)
    rel_csvs     =  create_relationship_csvs(target_dir, rel_dfs)
    node_flags   =  build_node_flags(node_csvs)
    rel_flags    =  build_rel_flags(rel_csvs)
    return          build_import_command(node_flags, rel_flags)
    # fmt: on


def load_csv_file(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, low_memory=False)


def clean_data(primekg_data: pd.DataFrame) -> pd.DataFrame:
    primekg_data = primekg_data.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
    type_columns = ["relation", "display_relation", "x_type", "y_type"]
    primekg_data[type_columns] = primekg_data[type_columns].replace({" ": "_", "-": "_"}, regex=True)
    return primekg_data.replace({"/": "_or_"}, regex=True)


def build_nodes_by_type_dfs(primekg_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    primekg_x = primekg_data[["x_index", "x_id", "x_type", "x_name", "x_source"]].drop_duplicates()
    primekg_x = primekg_x.rename(lambda l: l.lstrip("x_"), axis="columns")
    primekg_x = primekg_x.sort_values(by=["index"])
    types = primekg_x["type"].unique()
    return {t: _extract_node_df_for_type(primekg_x, t) for t in types}


def _extract_node_df_for_type(df: pd.DataFrame, type: str) -> pd.DataFrame:
    df = df[df["type"] == type]
    df = df.rename(columns={"index": f"index:ID({type})"})
    df = df.drop(columns=["type"])
    return df


def build_relationship_dfs(primekg_data: pd.DataFrame, use_display_rel: bool) -> Dict[str, List[pd.DataFrame]]:
    rel_key = "display_relation" if use_display_rel else "relation"
    return dict(_build_relationship_dfs(primekg_data, rel_key))


def _build_relationship_dfs(primekg_data: pd.DataFrame, rel_key: str) -> Iterable[Tuple[str, List[pd.DataFrame]]]:
    for rel in primekg_data[rel_key].unique():
        sub_df = primekg_data[primekg_data[rel_key] == rel]
        sub_df = sub_df[["x_index", "x_type", "y_type", "y_index"]]
        type_pairs = set(zip(sub_df.x_type, sub_df.y_type))
        yield rel, list(_extract_relationship_dfs_for_each_type_pair(sub_df, type_pairs))


def _extract_relationship_dfs_for_each_type_pair(
    sub_df: pd.DataFrame, type_pairs: Set[Tuple[str, str]]
) -> Iterable[pd.DataFrame]:
    for t1, t2 in type_pairs:
        yield _extract_relationship_df_for_type_pair(sub_df, t1, t2)


def _extract_relationship_df_for_type_pair(df: pd.DataFrame, type1: str, type2: str) -> pd.DataFrame:
    df = df[(df["x_type"] == type1) & (df["y_type"] == type2)]
    df = df.rename(columns={"x_index": f":START_ID({type1})", "y_index": f":END_ID({type2})"})
    df = df.drop(columns=["x_type", "y_type"])
    return df


def create_node_csvs(directory: str, node_dfs: Dict[str, pd.DataFrame]) -> Iterable[Tuple[str, str]]:
    for node_type, node_data in node_dfs.items():
        yield node_type, _create_csv_from_df(directory, node_type, node_data, "node")


def create_relationship_csvs(directory: str, rel_dfs: Dict[str, List[pd.DataFrame]]) -> Iterable[Tuple[str, List[str]]]:
    for rel_type, rdl in rel_dfs.items():
        fns = [_create_csv_from_df(directory, rel_type, rel_data, "rel", i + 1) for i, rel_data in enumerate(rdl)]
        yield rel_type, fns


def _create_csv_from_df(
    directory: str,
    type: str,
    rel_data: pd.DataFrame,
    prefix: str,
    idx: Optional[int] = None,
) -> str:
    type = type.replace("/", "_").replace(" ", "_")
    fn = f"{prefix}_{type}{idx if idx is not None else ''}.csv"
    fn = os.path.join(directory, fn)
    rel_data.to_csv(fn, index=False)
    return fn


def build_node_flags(node_csvs: Iterable[Tuple[str, str]]) -> Iterable[str]:
    for node_type, csv_file in node_csvs:
        yield f'--nodes={node_type.replace(" ", "_")}={csv_file}'


def build_rel_flags(rel_csvs: Iterable[Tuple[str, List[str]]]) -> Iterable[str]:
    for rel_type, csv_files in rel_csvs:
        for csv_file in csv_files:
            yield f'--relationships={rel_type.replace(" ", "_")}="{csv_file}"'


def build_import_command(node_flags: Iterable[str], rel_flags: Iterable[str], db_name: str = "neo4j") -> str:
    sep = " \\\n"
    return (
        f"neo4j-admin database import full{sep}"
        f"{sep.join(node_flags)}{sep}"
        f"{sep.join(rel_flags)}{sep}"
        f"--trim-strings=true --id-type=integer --verbose {db_name}"
        # If the database into which you import does not exist prior to importing,
        # you must create it subsequently using "CREATE DATABASE <db_name>".
        # Note: This command requires the professional version of Neo4j. Or this hack:
        # https://stackoverflow.com/questions/60429947/error-occurs-when-creating-a-new-database-under-neo4j-4-0
    )


if __name__ == "__main__":
    primekg_csv = os.getenv("PRIMEKG_CSV")
    target_dir = os.getenv("PRIMEKG_CSVS_FOR_NEO4J")
    assert primekg_csv is not None and target_dir is not None
    use_display_rel = True

    cmd = extract_node_and_relationship_csvs(primekg_csv, target_dir, use_display_rel)
    print(cmd)
