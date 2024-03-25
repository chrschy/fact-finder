from typing import Any, Dict, List

from pydantic.v1 import BaseModel


class GraphQAChainOutput(BaseModel):
    question: str
    cypher_query: str
    graph_response: List[Dict[str, Any]]
    answer: str
    evidence_sub_graph: List[Dict[str, Any]]
    expanded_evidence_subgraph: List[Dict[str, Any]]
