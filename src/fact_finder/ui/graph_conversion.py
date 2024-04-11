import asyncio
import sys
from typing import Any, Dict, List

from fact_finder.utils import (
    get_triples_from_graph_result,
    graph_result_contains_triple,
)
from pydantic import BaseModel


class Node(BaseModel):
    id: int
    type: str
    name: str
    in_query: bool
    in_answer: bool


class Edge(BaseModel):
    id: int
    type: str
    name: str
    source: int
    target: int
    in_query: bool
    in_answer: bool


class Subgraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]



def convert_subgraph(graph: List[Dict[str, Any]], result: List[Dict[str, Any]]) -> (Subgraph, str):
    graph_converted = Subgraph(nodes=[], edges=[])
    graph_triplets = ""

    try:
        result_ents = []
        for res in result:
            result_ents += res.values()

        idx_rel = 0
        for entry in graph:
            if graph_result_contains_triple(graph_result_entry=entry):
                graph_triplet, idx_rel = _process_triples(entry, graph_converted, result_ents, idx_rel)
            else:
                graph_triplet = _process_nodes_only(entry, graph_converted, result_ents)

            graph_triplets += graph_triplet

    except Exception as e:
        print(e)

    return (graph_converted, graph_triplets)


def _process_triples(entry, graph_converted: Subgraph, result_ents: list, idx_rel: int) -> int:
    triples_as_string = ""
    triples = get_triples_from_graph_result(graph_result_entry=entry)
    for triple in triples:
        triples_as_string += _process_triple(
            entry=entry, graph_converted=graph_converted, result_ents=result_ents, idx_rel=idx_rel, triple=triple
        )
        idx_rel += 1
    return triples_as_string, idx_rel


def _process_triple(entry, graph_converted: Subgraph, result_ents: list, idx_rel: int, triple: dict) -> str:
    graph_triplet = ""

    head_type = [key for key, value in entry.items() if value == triple[0]]
    tail_type = [key for key, value in entry.items() if value == triple[2]]
    head_type = head_type[0] if len(head_type) > 0 else ""
    tail_type = tail_type[0] if len(tail_type) > 0 else ""
    node_head = triple[0] if "index" in triple[0] else list(entry.values())[0]
    node_tail = triple[2] if "index" in triple[2] else list(entry.values())[2]

    if "index" in node_head and node_head["index"] not in [node.id for node in graph_converted.nodes]:
        graph_converted.nodes.append(
            Node(
                id=node_head["index"],
                type=head_type,
                name=node_head["name"],
                in_query=False,
                in_answer=node_head["name"] in result_ents,
            )
        )
    if "index" in node_tail and node_tail["index"] not in [node.id for node in graph_converted.nodes]:
        graph_converted.nodes.append(
            Node(
                id=node_tail["index"],
                type=tail_type,
                name=node_tail["name"],
                in_query=False,
                in_answer=node_tail["name"] in result_ents,
            )
        )
    if "index" in node_head and "index" in node_tail:
        graph_converted.edges.append(
            Edge(
                id=idx_rel,
                type=triple[1],
                name=triple[1],
                source=node_head["index"],
                target=node_tail["index"],
                in_query=False,
                in_answer=node_tail["name"] in result_ents,
            )
        )

    try:
        graph_triplet = f'("{node_head["name"]}", "{triple[1]}", "{node_tail["name"]}"), '
    except Exception as e:
        print(e)

    return graph_triplet


def _process_nodes_only(entry, graph_converted: Subgraph, result_ents: list) -> None:
    graph_triplet = ""
    for variable_binding, possible_node in entry.items():
        if not isinstance(possible_node, dict):
            continue
        if "index" in possible_node and possible_node["index"] not in [node.id for node in graph_converted.nodes]:
            graph_converted.nodes.append(
                Node(
                    id=possible_node["index"],
                    type=variable_binding,
                    name=possible_node["name"],
                    in_query=False,
                    in_answer=possible_node["name"] in result_ents,
                )
            )
            try:
                graph_triplet += f'("{possible_node["name"]}"), '
            except Exception as e:
                print(e)
    return graph_triplet

