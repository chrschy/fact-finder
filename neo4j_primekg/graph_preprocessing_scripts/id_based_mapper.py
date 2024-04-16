from abc import abstractmethod
from typing import List, Optional, Set, Tuple, Union

try:
    import polars as pd
except:
    import pandas as pd

from mapper import Mapper


class IdBasedMapper(Mapper):
    @abstractmethod
    def _get_relevant_node_types(self) -> Set[str]:
        """Create a list of node types to which this mapping is applied.

        :return: List of ids.
        :rtype: List[str]
        """

    @abstractmethod
    def _get_mapping_for_id(self, graph_id: str) -> Tuple[str, Optional[str]]:
        """Generates the corresponding id and preferred term/name for a given id.

        :param graph_id: Id in the base graph.
        :type graph_id: str
        :return: The new id and the new name. If no mapping exists, (graph_id, None) is returned for.
        :rtype: Tuple[str, Optional[str]]
        """
        ...

    def _call_polars(self, row: Tuple[str, ...]) -> Tuple[str, ...]:
        row_lst = list(row)
        row_lst = self._apply_mapping(row_lst)
        return tuple(row_lst)

    def _call_pandas(self, row: "pd.Series") -> "pd.Series":
        return self._apply_mapping(row)

    def _apply_mapping(self, row: Union["pd.Series", List[str]]) -> Union["pd.Series", List[str]]:
        node_type, id_in_graph = row[self._x_type_idx], row[self._x_id_idx]
        if node_type in self._get_relevant_node_types():
            new_id, new_name = self._get_mapping_for_id(id_in_graph)
            if new_name is not None:
                row[self._x_name_idx] = new_name
                row[self._x_id_idx] = new_id
        node_type, id_in_graph = row[self._y_type_idx], row[self._y_id_idx]
        if node_type in self._get_relevant_node_types():
            new_id, new_name = self._get_mapping_for_id(id_in_graph)
            if new_name is not None:
                row[self._y_name_idx] = new_name
                row[self._y_id_idx] = new_id
        return row
