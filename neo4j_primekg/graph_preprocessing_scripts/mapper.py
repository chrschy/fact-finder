from abc import ABC, abstractmethod
from typing import Tuple, Union

from tqdm import tqdm

_POLARS_AVAILABLE = False
try:
    import polars as pd

    _POLARS_AVAILABLE = True
except:
    import logging

    import pandas as pd

    logging.warning("Polars not available for CSV processing. Using pandas which might be slow.")


class Mapper(ABC):

    def __init__(self, graph: pd.DataFrame, use_polars: bool = _POLARS_AVAILABLE) -> None:
        self._use_polars = use_polars
        self._set_column_accessors(graph)

    def apply_to_graph(self, graph: pd.DataFrame) -> pd.DataFrame:
        if self._use_polars:
            col_names = graph.columns
            graph = graph.map_rows(self)
            graph.columns = col_names
        else:
            tqdm.pandas()
            graph = graph.progress_apply(self, axis="columns", result_type="broadcast")
        return graph

    def __call__(self, row: Union["pd.Series", Tuple[str, ...]]) -> Union["pd.Series", Tuple[str]]:
        if self._use_polars:
            return self._call_polars(row)
        return self._call_pandas(row)

    @abstractmethod
    def _call_polars(self, row: Tuple[str, ...]) -> Tuple[str, ...]: ...

    @abstractmethod
    def _call_pandas(self, row: "pd.Series") -> "pd.Series": ...

    def _set_column_accessors(self, df: pd.DataFrame):
        if self._use_polars:
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
