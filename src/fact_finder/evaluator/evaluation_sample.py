from pydantic import BaseModel
from typing import List, Dict


class EvaluationSample(BaseModel):
    question: str
    cypher_query: str
    sub_graph: List[Dict]
    question_is_answerable: bool
    source: str
    expected_answer: str
