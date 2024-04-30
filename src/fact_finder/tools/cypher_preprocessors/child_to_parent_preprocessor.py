from typing import Dict, List

import regex as re
from fact_finder.tools.cypher_preprocessors.property_string_preprocessor import (
    PropertyStringCypherQueryPreprocessor,
)
from langchain_community.graphs import Neo4jGraph


class ChildToParentPreprocessor(PropertyStringCypherQueryPreprocessor):
    _mapping_query = (
        "MATCH (child)-[:{child_to_parent_relation}]->(parent)\n"
        "WITH child, parent, COUNT{(child)-->()} AS outdegree\n"
        "WHERE outdegree = 1\n"
        "RETURN DISTINCT child.{name}, parent.{name}"
    )

    def __init__(
        self,
        graph: Neo4jGraph,
        child_to_parent_relation: str,
        name_property: str = "name",
    ) -> None:
        super().__init__(property_names=[name_property])
        self._name_property = name_property
        graph_result = self._run_graph_query(graph, child_to_parent_relation)
        self._child_to_parent: Dict[str, str] = self._to_parent_child_dict(graph_result)

    def _run_graph_query(self, graph: Neo4jGraph, child_to_parent_relation: str) -> List[Dict[str, str]]:
        query = self._mapping_query.replace("{child_to_parent_relation}", child_to_parent_relation)
        query = query.replace("{name}", self._name_property)
        return graph.query(query)

    def _to_parent_child_dict(self, graph_result: List[Dict[str, str]]) -> Dict[str, str]:
        res: Dict[str, str] = dict()
        for r in graph_result:
            child_name: str = r[f"child.{self._name_property}"]
            parent_name: str = r[f"parent.{self._name_property}"]
            if child_name != parent_name:
                res[child_name] = parent_name
        return res

    def _replace_match(self, matches: re.Match[str]) -> str:
        assert matches.groups()
        block = matches.group(0)
        block_start, _ = matches.spans(0)[0]
        prev_end = 0
        new_block = ""
        for node_name, (start, end) in sorted(zip(matches.captures(1), matches.spans(1)), key=lambda x: x[1][0]):
            if parent_name := self._child_to_parent.get(node_name):
                new_block += block[prev_end : start - block_start] + parent_name
                prev_end = end - block_start
        new_block += block[prev_end:]
        return new_block
