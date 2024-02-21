import ssl
from abc import ABC, abstractmethod
from typing import List

from nltk.corpus import wordnet as wn
from SPARQLWrapper import JSON, SPARQLWrapper


class SynonymFinder(ABC):

    @abstractmethod
    def __call__(self, name: str) -> List[str]:
        pass


class WikiDataSynonymFinder(SynonymFinder):

    def __init__(
        self,
        endpoint_url: str = "https://query.wikidata.org/sparql",
        user_agent: str = "factfinder/1.0",
    ):
        self.__endpoint_url = endpoint_url
        self.__user_agent = user_agent

    def __call__(self, name: str) -> List[str]:
        query = self.__generate_sparql_forwards_query(name)
        results = self.__get_sparql_results(query)
        if not results:
            query = self.__generate_sparql_backwards_query(name)
            results = self.__get_sparql_results(query)
        return results

    def __get_sparql_results(self, query):
        ssl._create_default_https_context = ssl._create_unverified_context
        sparql = SPARQLWrapper(self.__endpoint_url, agent=self.__user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        results = [result["label"]["value"] for result in results["results"]["bindings"]]
        return results

    def __generate_sparql_forwards_query(self, name: str) -> str:
        query = """SELECT DISTINCT ?label WHERE {
                  ?s skos:altLabel "%s"@en.
                  ?s rdfs:label ?label .
                  FILTER(str(lang(?label)) = "en")
                }""" % (
            name
        )
        return query

    def __generate_sparql_backwards_query(self, name: str) -> str:
        # todo test and adjust the query such that it works when node="alcohol" -> "ethanol"
        query = """SELECT DISTINCT ?alt_label WHERE {
                ?s skos:altLabel ?alt_label .
                ?s rdfs:label "%s"@en .
                FILTER(str(lang(?label)) = "en")
                }""" % (
            name
        )
        return query


class WordNetSynonymFinder(SynonymFinder):

    def __call__(self, name: str) -> List[str]:
        result = wn.synonyms(word=name)[0]
        return result


class SimilaritySynonymFinder(SynonymFinder):

    def __call__(self, name: str) -> List[str]:
        # todo find most similar node from all nodes with vector cosine similarity search
        raise NotImplementedError


class SubWordSynonymFinder(SynonymFinder):

    def __call__(self, name: str) -> List[str]:
        # todo 2010.11784.pdf (arxiv.org)
        raise NotImplementedError
