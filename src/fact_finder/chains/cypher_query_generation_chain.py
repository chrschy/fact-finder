from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chains.graph_qa.cypher import construct_schema, extract_cypher
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate

from fact_finder.predicate_descriptions import PREDICATE_DESCRIPTIONS


class CypherQueryGenerationChain(Chain):
    cypher_generation_chain: LLMChain
    graph_schema: str
    n_predicate_descriptions: int
    return_intermediate_steps: bool
    input_key: str = "question"  #: :meta private:
    output_key: str = "cypher_query"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"

    def __init__(
        self,
        llm: BaseLanguageModel,
        prompt_template: BasePromptTemplate,
        graph_structured_schema: Dict[str, Any],
        n_predicate_descriptions: int = 0,
        return_intermediate_steps: bool = True,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
    ):
        cypher_generation_chain = LLMChain(llm=llm, prompt=prompt_template)
        if exclude_types and include_types:
            raise ValueError("Either `exclude_types` or `include_types` " "can be provided, but not both")
        graph_schema = construct_schema(graph_structured_schema, include_types, exclude_types)
        super().__init__(
            cypher_generation_chain=cypher_generation_chain,
            graph_schema=graph_schema,
            n_predicate_descriptions=n_predicate_descriptions,
            return_intermediate_steps=return_intermediate_steps,
        )

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        generated_cypher = self._generate_cypher(question, _run_manager)
        return self._prepare_chain_result(inputs, generated_cypher)

    def _generate_cypher(self, question: str, run_manager: CallbackManagerForChainRun):
        predicate_descriptions = self._construct_predicate_descriptions(how_many=self.n_predicate_descriptions)
        generated_cypher = self.cypher_generation_chain(
            {"question": question, "schema": self.graph_schema, "predicate_descriptions": predicate_descriptions},
            callbacks=run_manager.get_child(),
        )[self.cypher_generation_chain.output_key]
        generated_cypher = extract_cypher(generated_cypher)
        self._log_it(generated_cypher, run_manager)
        return generated_cypher

    def _construct_predicate_descriptions(self, how_many: int) -> str:
        if how_many > 0:
            result = ["Here are some descriptions to the most common relationships:"]
            for item in PREDICATE_DESCRIPTIONS[:how_many]:
                item_as_text = f"({item['subject']})-[{item['predicate']}]->({item['object']}): {item['definition']}"
                result.append(item_as_text)
            return "\n".join(result)
        else:
            return ""

    def _log_it(self, generated_cypher: str, run_manager: CallbackManagerForChainRun):
        run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        run_manager.on_text(generated_cypher, color="green", end="\n", verbose=self.verbose)

    def _prepare_chain_result(self, inputs: Dict[str, Any], generated_cypher: str) -> Dict[str, Any]:
        chain_result = {self.output_key: generated_cypher}
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            intermediate_steps += [{"question": inputs[self.input_key]}, {self.output_key: generated_cypher}]
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result
