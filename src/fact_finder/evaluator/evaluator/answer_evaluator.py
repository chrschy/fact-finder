from typing import Dict

from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluator.evaluator import Evaluator


class AnswerEvaluator(Evaluator):

    def expected_response(self, evaluation_sample: EvaluationSample) -> str:
        return evaluation_sample.expected_answer

    def generated_response(self, system_response: Dict) -> str:
        return system_response["graph_qa_output"].answer
