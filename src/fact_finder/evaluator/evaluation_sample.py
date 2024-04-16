from pydantic import BaseModel
from typing import List, Dict, Any


class EvaluationSample(BaseModel):
    question: str
    cypher_query: str
    question_is_answerable: bool
    source: str
    expected_answer: str
    sub_graph: List[Dict] = []
    nodes: List[Dict[str, Any]] = []
