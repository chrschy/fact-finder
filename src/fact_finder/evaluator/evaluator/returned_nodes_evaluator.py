from typing import Dict

from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluator.evaluator import Evaluator


class ReturnedNodesEvaluator(Evaluator):
    def expected_response(self, evaluation_sample: EvaluationSample) -> str:
        raise NotImplementedError

    def generated_response(self, system_response: Dict) -> str:
        raise NotImplementedError
