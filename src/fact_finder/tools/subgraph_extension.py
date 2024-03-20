import copy
from typing import List, Dict, Any

from langchain_community.graphs import Neo4jGraph

from fact_finder.utils import graph_result_contains_triple, get_triples_from_graph_result


class SubgraphExpansion:

    def __init__(self, graph: Neo4jGraph):
        self._graph: Neo4jGraph = graph
        self._extension_query_template = """
        MATCH (a {{index: {head_index} }})-[r1]-(c)-[r2]-(b {{index: {tail_index} }})
        RETURN a, r1, c, r2, b
        """

    def expand(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nodes = copy.deepcopy(nodes)
        result = []
        for entry in nodes:
            if graph_result_contains_triple(graph_result_entry=entry):
                triples = get_triples_from_graph_result(graph_result_entry=entry)
                for triple in triples:
                    result += self._enrich(triple=triple)
        return nodes + result

    def _enrich(self, triple) -> List[Dict[str, Any]]:
        head_index = triple[0]["index"]
        tail_index = triple[2]["index"]
        extension_query = self._extension_query_template.format(head_index=head_index, tail_index=tail_index)
        return self._query_graph(cypher=extension_query)

    def _query_graph(self, cypher) -> List[Dict[str, Any]]:
        try:
            return self._graph.query(cypher)
        except Exception as e:
            print(f"Sub Graph for {cypher} could not be extracted due to {e}")
        return []
