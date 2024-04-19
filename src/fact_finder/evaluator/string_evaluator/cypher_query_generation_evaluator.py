from typing import Dict

from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.string_evaluator.string_evaluator import StringEvaluator


class CypherQueryGenerationStringEvaluator(StringEvaluator):

    def expected_response(self, evaluation_sample: EvaluationSample) -> str:
        return evaluation_sample.cypher_query

    def generated_response(self, system_response: Dict) -> str:
        return system_response["graph_qa_output"].cypher_query
