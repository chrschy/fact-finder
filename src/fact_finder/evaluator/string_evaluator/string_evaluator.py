from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Set

from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.score.score import Score


class StringEvaluator(ABC):
    QUESTION_KEY = "question"
    EXPECTED_RESPONSE_KEY = "expected_response"
    GENERATED_RESPONSE_KEY = "generated_response"

    @abstractmethod
    def expected_response(self, evaluation_sample: EvaluationSample) -> str: ...
    @abstractmethod
    def generated_response(self, system_response: Dict) -> str: ...

    def evaluate(
        self, evaluation_samples: List[EvaluationSample], chain_results: List[Dict[str, Any]], scores: List[Score]
    ) -> List[Dict[str, Any]]:
        assert len(evaluation_samples) == len(chain_results)
        eval_results = []
        for i, eval_sample in enumerate(evaluation_samples):
            system_response = chain_results[i]
            eval_result = self.construct_result(eval_sample, system_response)
            for score in scores:
                score_result = self.calculate_similarity(eval_sample, system_response, score)
                eval_result[score.__class__.__name__] = score_result
            eval_results.append(eval_result)
        return eval_results

    def calculate_similarity(self, evaluation_sample: EvaluationSample, system_response: Dict, score: Score) -> float:
        return score.compare(self.expected_response(evaluation_sample), self.generated_response(system_response))

    def construct_result(self, evaluation_sample: EvaluationSample, system_response: Dict) -> Dict[str, Any]:
        return {
            self.QUESTION_KEY: evaluation_sample.question,
            self.EXPECTED_RESPONSE_KEY: self.expected_response(evaluation_sample),
            self.GENERATED_RESPONSE_KEY: self.generated_response(system_response),
        }
