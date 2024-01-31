import os

import pytest
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

from fact_finder.synonym_selector.synonym_finder import WikiDataSynonymFinder, WordNetSynonymFinder
from fact_finder.synonym_selector.synonym_selector import SynonymSelector

load_dotenv()


@pytest.fixture(scope="session")
def graph():
    return Neo4jGraph(url=os.getenv("NEO4J_URL"), username=os.getenv("NEO4J_USER"), password=os.getenv("NEO4J_PW"))


@pytest.fixture()
def wiki_synonym_selector(graph):
    return SynonymSelector(graph=graph, synonym_finder=WikiDataSynonymFinder())


@pytest.fixture()
def wordnet_synonym_selector(graph):
    return SynonymSelector(graph=graph, synonym_finder=WordNetSynonymFinder())


def test_wiki_synonym_selector(wiki_synonym_selector):
    query = 'MATCH (e:exposure {name: "alcohol"})-[:linked_to]->(d:disease) RETURN d.name'
    selector_result = wiki_synonym_selector(query)
    assert selector_result == 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    query = 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    selector_result = wiki_synonym_selector(query)
    assert selector_result == 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'


def test_wordnet_synonym_selector(wordnet_synonym_selector):
    alcohol_query = 'MATCH (e:exposure {name: "alcohol"})-[:linked_to]->(d:disease) RETURN d.name'
    selector_result = wordnet_synonym_selector(alcohol_query)
    assert selector_result == 'MATCH (e:exposure {name: "alcohol"})-[:linked_to]->(d:disease) RETURN d.name'
