import re
import string
from abc import ABC, abstractmethod

from fact_finder.chains.custom_llm_chain import CustomLLMChain
from langchain_core.language_models import BaseLanguageModel

from fact_finder.prompt_templates import SUBGRAPH_EXTRACTOR_PROMPT


class SubGraphExtractor(ABC):
    @abstractmethod
    def __call__(self, cypher_query: str) -> str: ...


class LLMSubGraphExtractor(SubGraphExtractor):

    def __init__(self, model: BaseLanguageModel):
        self.llm_chain = CustomLLMChain(llm=model, prompt=SUBGRAPH_EXTRACTOR_PROMPT)

    def __call__(self, cypher_query: str) -> str:
        result = self.llm_chain(cypher_query)
        return result["text"]


class RegexSubGraphExtractor(SubGraphExtractor):

    def __init__(self):
        self.__extract_group_regex = r"(\(.*\))[-|<|>]+(\[.*\])[-|<|>]+(\(.*\))"
        self.__extract_subject_regex = self.__extract_object_regex = r"(?<=\()([a-zA-Z0-9]*)"
        self.__extract_predicate_regex = r"(?<=\[)([a-zA-Z0-9]*)"

    def __call__(self, cypher_query: str) -> str:
        result = re.search(self.__extract_group_regex, cypher_query)
        subject, predicate, object = result.groups()
        subject_variable = re.search(self.__extract_subject_regex, subject).group(0)
        predicate_variable = re.search(self.__extract_predicate_regex, predicate).group(0)
        object_variable = re.search(self.__extract_object_regex, object).group(0)
        all_letters = list(string.ascii_lowercase)
        all_possible_letters = list(set(all_letters) - {object_variable, subject_variable, predicate_variable})
        if len(subject_variable) == 0:
            subject_variable = all_possible_letters.pop()
            new_subject = self.__replace_with_variable(subject, subject_variable)
            cypher_query = cypher_query.replace(subject, new_subject)
        if len(predicate_variable) == 0:
            predicate_variable = all_possible_letters.pop()
            new_predicate = self.__replace_with_variable(predicate, predicate_variable)
            cypher_query = cypher_query.replace(predicate, new_predicate)
        if len(object_variable) == 0:
            object_variable = all_possible_letters.pop()
            new_object = self.__replace_with_variable(object, object_variable)
            cypher_query = cypher_query.replace(object, new_object)
        cypher_query = re.sub(
            r"return.*",
            f"return {subject_variable},{predicate_variable},{object_variable}",
            cypher_query,
            flags=re.IGNORECASE,
        )
        return cypher_query

    def __replace_with_variable(self, text_without_variable: str, variable: str) -> str:
        result = text_without_variable[0] + variable + text_without_variable[1:]
        return result
