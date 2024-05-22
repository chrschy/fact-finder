from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from fact_finder.tools.entity_detector import EntityDetector
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun


class EntityDetectionQuestionPreprocessingChain(Chain):
    entity_detector: EntityDetector
    # The keys of this dict contain the (lower case) type names for which entities can be replaced.
    # They map to a template explaining the type of an entity (marked via {entity})
    # Example: "chemical_compounds", "{entity} is a chemical compound."
    allowed_types_and_description_templates: Dict[str, str]
    return_intermediate_steps: bool = True
    input_key: str = "question"  #: :meta private:
    output_key: str = "preprocessed_question"  #: :meta private:
    intermediate_steps_key: str = "intermediate_steps"

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
        run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]

        entity_results = self._compute_entities_sorted_by_interval_start(question)
        if len(entity_results) == 0:
            return self._prepare_chain_result(inputs, entity_results, question)
        entity_results = self._clear_overlapping_intervals(entity_results, len(question))
        self._log_it(entity_results, run_manager)
        new_question, entity_type_hints = self._extract_entity_data(question, entity_results)
        new_question = new_question + "\nAdditional Information:\n" + "\n".join(entity_type_hints)

        return self._prepare_chain_result(inputs, entity_results, new_question)

    def _compute_entities_sorted_by_interval_start(self, question):
        entity_results = sorted(
            [
                (r["start_span"], r["end_span"], r["pref_term"], r["sem_type"])
                for r in self.entity_detector(question)
                if r["sem_type"].lower() in self.allowed_types_and_description_templates
            ],
            key=lambda x: (x[0], -x[1]),
            # Sort by start_span, ascending first and end_span, descending second.
            # With this, subset intervals appear after the larger interval.
        )

        return entity_results

    def _clear_overlapping_intervals(
        self, entity_results: List[Tuple[int, int, str, str]], len_str: int
    ) -> List[Tuple[int, int, str, str]]:
        last_end = 0
        last_idx: Optional[int] = None
        filtered_entity_results: List[Tuple[int, int, str, str]] = []
        for i, entity in enumerate(entity_results):
            start, end = entity[:2]
            last_end, last_idx = self._test_and_handle_overlap_with_previous_interval(
                entity_results, last_end, last_idx, filtered_entity_results, i, start, end
            )
        self._test_and_handle_overlap_with_previous_interval(
            entity_results, last_end, last_idx, filtered_entity_results, i + 1, len_str, len_str
        )
        return filtered_entity_results

    def _test_and_handle_overlap_with_previous_interval(
        self,
        entity_results: List[Tuple[int, int, str, str]],
        last_end: int,
        last_idx: Optional[int],
        filtered_entity_results: List[Tuple[int, int, str, str]],
        current_index: int,
        start: int,
        end: int,
    ) -> Tuple[int, Optional[int]]:
        if last_end <= start:
            # If no overlap exists, the previous interval can be added
            # to the results and the current as the next candidate.
            if last_idx is not None:
                filtered_entity_results.append(entity_results[last_idx])
            last_end = end
            last_idx = current_index
        elif end > last_end:
            # If current entity is contained in previous one, it will be ignored/removed.
            # Partial overlap is unclear and cannot be resolved. Then, both are skipped.
            last_end = end
            last_idx = None
        return last_end, last_idx

    def _log_it(self, entity_results: List[Tuple[int, int, str, str]], run_manager: CallbackManagerForChainRun):
        run_manager.on_text("Entities detected in question:", end="\n", verbose=self.verbose)
        entities = ", ".join([f"{e[2]} ({e[3]})" for e in entity_results])
        run_manager.on_text(entities, color="green", end="\n", verbose=self.verbose)

    def _extract_entity_data(
        self, question: str, entity_results: List[Tuple[int, int, str, str]]
    ) -> Tuple[str, List[str]]:
        entity_type_hints = []
        new_question = ""
        last_index = 0
        for start, end, pref_name, type in entity_results:
            new_question += question[last_index:start] + pref_name
            last_index = end
            entity_type_hints.append(self._create_type_hint(pref_name, type))
        new_question += question[last_index:]
        return new_question, entity_type_hints

    def _create_type_hint(self, preferred_name: str, entity_type: str) -> str:
        entity_type = entity_type.lower()
        template = self.allowed_types_and_description_templates[entity_type.lower()]
        hint = template.replace("{entity}", preferred_name)
        hint = hint[0].upper() + hint[1:]
        return hint

    def _prepare_chain_result(
        self, inputs: Dict[str, Any], entity_results: List[Tuple[str, str]], new_question: str
    ) -> Dict[str, Any]:
        chain_result = {self.output_key: new_question}
        if self.return_intermediate_steps:
            intermediate_steps = inputs.get(self.intermediate_steps_key, [])
            intermediate_steps += [{"original_question": inputs[self.input_key]}]
            intermediate_steps += [{"entity_results": entity_results}]
            intermediate_steps += [{self.output_key: new_question}]
            chain_result[self.intermediate_steps_key] = intermediate_steps
        return chain_result


@runtime_checkable
class EntityDetectionQuestionPreprocessingProtocol(Protocol):
    def __call__(
        self,
        *,
        entity_detector: EntityDetector,
        allowed_types_and_description_templates: Dict[str, str],
        return_intermediate_steps: bool = True,
    ) -> EntityDetectionQuestionPreprocessingChain: ...
