from typing import Any, Dict, List

from pydantic import BaseModel


class EvaluationSample(BaseModel):
    question: str
    cypher_query: str
    expected_answer: str = ""
    nodes: List[Dict[str, Any]] = []
