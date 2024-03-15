import os
import sys
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

T = TypeVar("T")
# P = ParamSpec("P")


def log_it(fnct: Callable[..., T]) -> Callable[..., T]:
    name = fnct.__name__.strip("_").replace("_", " ")

    def _fnct_with_log(*args: "P.args", **kwargs: "P.kwargs") -> T:
        print(f'Executing step "{name}"', file=sys.stderr, flush=True)
        for k, v in kwargs.items():
            if isinstance(v, str):
                print(f'    Argument "{k}" has value: {v}', file=sys.stderr, flush=True)

        return fnct(*args, **kwargs)

    return _fnct_with_log


@log_it
def load_csv_file(filename: Optional[str], delimiter: Optional[str] = None) -> Optional[pd.DataFrame]:
    if filename and os.path.isfile(filename):
        return pd.read_csv(filename, delimiter=delimiter, low_memory=False)
    return None


class Neo4jPrimeKGImporter:
    _property_drug_feature_columns = [
        "node_index",
        "description",
        "half_life",
        # "indication",  # There already is an "indication" relation in the graph.
        "mechanism_of_action",
        # "protein_binding",  # Might be covered by "target" relation in the graph.
        "pharmacodynamics",
        "state",
        # "atc_1",
        # "atc_2",
        # "atc_3",
        # "atc_4",
        # "category",  # Handled as new nodes in the graph.
        # "group",  # Handled as new nodes in the graph.
        # "pathway",  # There are already "pathway" nodes in the graph.
        "molecular_weight",
        "tpsa",
        "clogp",
    ]
    _property_disease_feature_columns = [
        "node_index",
        # "mondo_id",
        # "mondo_name",
        # "group_id_bert",
        # "group_name_bert",
        # "mondo_definition",
        # "umls_description",
        # "orphanet_definition",
        "orphanet_prevalence",
        "orphanet_epidemiology",
        # "orphanet_clinical_description",
        "orphanet_management_and_treatment",
        # "mayo_symptoms",  # symptom description can conflict with phenotype relation that is already part of PrimeKG
        "mayo_causes",
        "mayo_risk_factors",
        "mayo_complications",
        "mayo_prevention",
        "mayo_see_doc",
        "description",
    ]

    def __init__(
        self,
        primekg_csv: str,
        target_dir: str,
        drug_features_tsv: Optional[str] = None,
        disease_features_tsv: Optional[str] = None,
        use_display_relation: bool = True,
        db_name: str = "neo4j",
    ) -> None:
        self._primekg_data: pd.DataFrame = load_csv_file(filename=primekg_csv)
        self._drug_features = load_csv_file(filename=drug_features_tsv, delimiter="\t")
        self._disease_features = load_csv_file(filename=disease_features_tsv, delimiter="\t")
        self._target_dir = target_dir
        self._use_display_relation = use_display_relation
        self._db_name = db_name
        self._available_start_index = self._primekg_data["x_index"].max()
        self._node_data_frames: Dict[str, pd.DataFrame] = {}
        self._relation_data_frames: Dict[str, List[pd.DataFrame]] = {}

    def __call__(self):
        self._clean_data()
        self._build_nodes_by_type_dfs()
        self._build_relationship_dfs()
        self._build_drug_category_relation()
        self._build_drug_approval_status_relation()
        node_csvs = self._create_node_csvs()
        rel_csvs = self._create_relationship_csvs()
        node_flags = self._build_node_flags(node_csvs)
        rel_flags = self._build_rel_flags(rel_csvs)
        return self._build_import_command(node_flags=node_flags, rel_flags=rel_flags)

    @log_it
    def _clean_data(self):
        for column in self._primekg_data.columns:
            if self._primekg_data[column].dtype == "object":
                self._primekg_data[column] = self._primekg_data[column].str.lower()
        type_columns = ["relation", "display_relation", "x_type", "y_type"]
        self._primekg_data[type_columns] = self._primekg_data[type_columns].apply(
            lambda x: x.str.replace(" ", "_").str.replace("-", "_")
        )
        self._primekg_data.replace("/", "_or_", regex=True, inplace=True)
        if self._drug_features is not None:
            self._drug_features = self._cleanup_undesired_symbols(self._drug_features)
        if self._disease_features is not None:
            self._disease_features = self._cleanup_undesired_symbols(self._disease_features)

    def _cleanup_undesired_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        df.replace("\r", "", regex=True, inplace=True)
        df.replace("\n", " ", regex=True, inplace=True)
        df.replace(",,", ",", regex=True, inplace=True)
        df.replace('"', "", regex=True, inplace=True)
        return df

    @log_it
    def _build_nodes_by_type_dfs(self):
        df = self._primekg_data[["x_index", "x_id", "x_type", "x_name", "x_source"]].drop_duplicates()
        df = df.rename(lambda l: l.lstrip("x_"), axis="columns")
        df = df.sort_values(by=["index"])
        types = df["type"].unique()
        for t in types:
            self._node_data_frames[t] = self._extract_node_df_for_type(df, t)

    def _extract_node_df_for_type(self, df: pd.DataFrame, type: str) -> pd.DataFrame:
        df = df[df["type"] == type]
        if type.lower() == "drug":
            df = self._add_additional_drug_fields(df)
        elif type.lower() == "disease":
            df = self._add_additional_disease_fields(df)
        df = df.rename(columns={"index": f"index:ID({type})"})
        df = df.drop(columns=["type"])
        return df

    @log_it
    def _add_additional_drug_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._drug_features is None:
            return df
        node_df = self._drug_features[self._property_drug_feature_columns].copy()
        node_df = self._extract_drug_node_property_values(node_df)
        node_df = node_df.rename(columns={"node_index": "index", "state": "aggregate_state"})
        string_columns = node_df.select_dtypes(include=[object])
        node_df[string_columns.columns] = string_columns.fillna("no information available")
        node_df = node_df.rename(
            columns={"molecular_weight": "molecular_weight:float", "tpsa": "tpsa:float", "clogp": "clogp:float"}
        )
        return pd.merge(df, node_df, on="index")

    def _extract_drug_node_property_values(self, node_df: pd.DataFrame) -> pd.DataFrame:
        node_df["state"] = node_df["state"].apply(
            lambda s: s.split(" ")[-1].strip(".") if isinstance(s, str) else s,
        )
        node_df["molecular_weight"] = node_df["molecular_weight"].apply(
            lambda s: float(s[len("The molecular weight is ") : -1]) if isinstance(s, str) else s
        )
        node_df["tpsa"] = node_df["tpsa"].apply(
            lambda s: float(s.split(" ")[-1].strip(".")) if isinstance(s, str) else s
        )
        node_df["clogp"] = node_df["clogp"].apply(
            lambda s: float(s.split(" ")[-1].strip(".")) if isinstance(s, str) else s
        )
        return node_df

    @log_it
    def _add_additional_disease_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._disease_features is None:
            return df
        node_df = self._disease_features.groupby("node_index").first().reset_index()
        node_df = self._extract_disease_node_description(node_df)
        node_df = node_df[self._property_disease_feature_columns]
        node_df = self._extract_disease_node_property_values(node_df)
        node_df = self._rename_disease_property_columns(node_df)
        string_columns = node_df.select_dtypes(include=[object])
        node_df[string_columns.columns] = string_columns.fillna("no information available")
        return pd.merge(df, node_df, on="index")

    def _extract_disease_node_description(self, node_df: pd.DataFrame) -> pd.DataFrame:
        desc_keys_by_priority = [
            "orphanet_clinical_description",
            "mondo_definition",
            "umls_description",
            "orphanet_definition",
        ]
        col: Union[pd.Series[str], np.NDArray[str]] = node_df[desc_keys_by_priority[-1]]
        for i in range(-2, -len(desc_keys_by_priority) - 1, -1):
            col = np.where(node_df[desc_keys_by_priority[i]].isna(), col, node_df[desc_keys_by_priority[i]])
        node_df["description"] = col
        return node_df

    def _extract_disease_node_property_values(self, node_df: pd.DataFrame) -> pd.DataFrame:
        node_df["mayo_see_doc"] = node_df["mayo_see_doc"].apply(
            lambda s: s.replace("When to see a doctor, ", "") if isinstance(s, str) else s
        )
        return node_df

    def _rename_disease_property_columns(self, node_df: pd.DataFrame) -> pd.DataFrame:
        return node_df.rename(
            columns={
                "node_index": "index",
                "orphanet_prevalence": "prevalence",
                "orphanet_epidemiology": "epidemiology",
                "orphanet_management_and_treatment": "management_and_treatment",
                # "mayo_symptoms": "symptoms",
                "mayo_causes": "causes",
                "mayo_risk_factors": "risk_factors",
                "mayo_complications": "complications",
                "mayo_prevention": "prevention",
                "mayo_see_doc": "when_to_see_a_doctor",
            }
        )

    @log_it
    def _build_relationship_dfs(self):
        rel_key = "display_relation" if self._use_display_relation else "relation"
        for rel in self._primekg_data[rel_key].unique():
            sub_df = self._primekg_data[self._primekg_data[rel_key] == rel]
            sub_df = sub_df[["x_index", "x_type", "y_type", "y_index"]]
            type_pairs = set(zip(sub_df.x_type, sub_df.y_type))
            rel_dfs = list(self._extract_relationship_dfs_for_each_type_pair(sub_df, type_pairs))
            self._relation_data_frames[rel] = rel_dfs

    def _extract_relationship_dfs_for_each_type_pair(
        self, sub_df: pd.DataFrame, type_pairs: Set[Tuple[str, str]]
    ) -> Iterable[pd.DataFrame]:
        for t1, t2 in type_pairs:
            yield self._extract_relationship_df_for_type_pair(sub_df, t1, t2)

    def _extract_relationship_df_for_type_pair(self, df: pd.DataFrame, type1: str, type2: str) -> pd.DataFrame:
        df = df[(df["x_type"] == type1) & (df["y_type"] == type2)]
        df = df.rename(columns={"x_index": f":START_ID({type1})", "y_index": f":END_ID({type2})"})
        df = df.drop(columns=["x_type", "y_type"])
        return df

    def _build_drug_category_relation(self):
        key = "category"
        new_key = "category"
        final_text = " is part of "
        seperator = " ; "
        rel_name = "is_a"
        self._build_drug_feature_relation(key, new_key, final_text, seperator, rel_name)

    def _build_drug_approval_status_relation(self):
        key = "group"
        new_key = "approval_status"
        final_text = " is "
        seperator = " and "
        rel_name = "has"
        self._build_drug_feature_relation(key, new_key, final_text, seperator, rel_name)

    def _build_drug_feature_relation(self, key: str, new_key: str, final_text: str, seperator: str, rel_name: str):
        if self._drug_features is None:
            return
        row_proc_args = (key, final_text, seperator)
        category_rel_df = self._drug_features[["node_index", key]].dropna()
        processed_rows = category_rel_df.apply(self._process_feat_row, axis=1, args=row_proc_args)
        category_rel_df = pd.concat(processed_rows.tolist(), ignore_index=True)
        self._build_feature_node_df(category_rel_df, key=key, new_key=new_key)
        category_rel_df = pd.merge(
            category_rel_df, self._node_data_frames[new_key], left_on=key, right_on="name", how="left"
        )
        category_rel_df = category_rel_df.rename(
            columns={"node_index": ":START_ID(drug)", f"index:ID({new_key})": f":END_ID({new_key})"}
        )
        self._relation_data_frames[rel_name] = [category_rel_df.drop(columns=[key, "name"])]

    def _process_feat_row(self, row: pd.Series, key: str, final_text: str, sep: str) -> pd.DataFrame:
        all_entries_for_node = row[key].split(final_text)[-1][:-1].split(sep)
        new_rows = {
            "node_index": [row["node_index"] for _ in all_entries_for_node],
            key: all_entries_for_node,
        }
        return pd.DataFrame(new_rows)

    def _build_feature_node_df(self, pre_relation_df: pd.DataFrame, key: str, new_key: str):
        node_names = pre_relation_df[key].unique()
        node_df = pd.DataFrame({"name": node_names})
        index_col_name = f"index:ID({new_key})"
        node_df[index_col_name] = node_df.index + self._available_start_index

        self._node_data_frames[new_key] = node_df
        self._available_start_index = node_df[index_col_name].max() + 1

    @log_it
    def _create_node_csvs(self) -> Iterable[Tuple[str, str]]:
        for node_type, node_data in self._node_data_frames.items():
            yield node_type, self._create_csv_from_df(node_type, node_data, "node")

    @log_it
    def _create_relationship_csvs(self) -> Iterable[Tuple[str, List[str]]]:
        for rel_type, rdl in self._relation_data_frames.items():
            fns = [self._create_csv_from_df(rel_type, rel_data, "rel", i + 1) for i, rel_data in enumerate(rdl)]
            yield rel_type, fns

    def _create_csv_from_df(
        self,
        type: str,
        rel_data: pd.DataFrame,
        prefix: str,
        idx: Optional[int] = None,
    ) -> str:
        type = type.replace("/", "_").replace(" ", "_")
        fn = f"{prefix}_{type}{idx if idx is not None else ''}.csv"
        fn = os.path.join(self._target_dir, fn)
        rel_data.to_csv(fn, index=False)
        return fn

    @log_it
    def _build_node_flags(self, node_csvs: Iterable[Tuple[str, str]]) -> Iterable[str]:
        for node_type, csv_file in node_csvs:
            yield f'--nodes={node_type.replace(" ", "_")}={csv_file}'

    @log_it
    def _build_rel_flags(self, rel_csvs: Iterable[Tuple[str, List[str]]]) -> Iterable[str]:
        for rel_type, csv_files in rel_csvs:
            for csv_file in csv_files:
                yield f'--relationships={rel_type.replace(" ", "_")}="{csv_file}"'

    @log_it
    def _build_import_command(self, node_flags: Iterable[str], rel_flags: Iterable[str]) -> str:
        sep = " \\\n"
        return (
            f"neo4j-admin database import full{sep}"
            f"{sep.join(node_flags)}{sep}"
            f"{sep.join(rel_flags)}{sep}"
            f"--trim-strings=true --id-type=integer --verbose {self._db_name}"
            # If the database into which you import does not exist prior to importing,
            # you must create it subsequently using "CREATE DATABASE <db_name>".
            # Note: This command requires the professional version of Neo4j. Or this hack:
            # https://stackoverflow.com/questions/60429947/error-occurs-when-creating-a-new-database-under-neo4j-4-0
        )


if __name__ == "__main__":
    primekg_csv = os.getenv("PRIMEKG_CSV")
    drug_features_tsv = os.getenv("PRIMEKG_DRUG_NODE_FEATURES_TSV", None)
    disease_features_tsv = os.getenv("PRIMEKG_DISEASE_NODE_FEATURES_TSV", None)
    target_dir = os.getenv("PRIMEKG_CSVS_FOR_NEO4J")
    assert primekg_csv is not None and os.path.isfile(primekg_csv)
    assert target_dir is not None
    assert drug_features_tsv is not None and os.path.isfile(drug_features_tsv)
    assert disease_features_tsv is not None and os.path.isfile(disease_features_tsv)
    use_display_rel = True

    os.makedirs(target_dir, exist_ok=True)

    importer = Neo4jPrimeKGImporter(
        primekg_csv=primekg_csv,
        target_dir=target_dir,
        drug_features_tsv=drug_features_tsv,
        disease_features_tsv=disease_features_tsv,
        use_display_relation=use_display_rel,
    )

    print(f"Starting graph extraction from csv file.", file=sys.stderr, flush=True)
    cmd = importer()
    print(f"Finished graph extraction from csv file.", file=sys.stderr, flush=True)
    print(cmd)
