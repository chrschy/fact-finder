from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

from fact_finder.qa_service.cypher_preprocessors.synonym_cypher_query_preprocessor import SynonymCypherQueryPreprocessor
from fact_finder.synonym_finder.synonym_finder import WikiDataSynonymFinder, WordNetSynonymFinder

load_dotenv()


@pytest.fixture(scope="session")
def graph():
    graph = MagicMock(spec=Neo4jGraph)
    graph.query = MagicMock()
    graph.query.return_value = [{"n": {"name": "ethanol"}}]
    return graph


@pytest.fixture()
def wiki_synonym_preprocessor(graph):
    synonym_finder = MagicMock(spec=WikiDataSynonymFinder)
    synonym_finder.find = MagicMock()
    synonym_finder.find.return_value = ["ethanol", "alcohols", "alcohol by volume"]
    return SynonymCypherQueryPreprocessor(graph=graph, synonym_finder=synonym_finder)


@pytest.fixture()
def wordnet_synonym_preprocessor(graph):
    synonym_finder = MagicMock(spec=WikiDataSynonymFinder)
    synonym_finder.find = MagicMock()
    synonym_finder.find.return_value = ["alcoholic_beverage", "alcoholic_drink", "inebriant", "intoxicant"]
    return SynonymCypherQueryPreprocessor(graph=graph, synonym_finder=synonym_finder)


def test_wiki_synonym_selector(wiki_synonym_preprocessor):
    query = 'MATCH (e:exposure {name: "alcohol"})-[:linked_to]->(d:disease) RETURN d.name'
    selector_result = wiki_synonym_preprocessor(query)
    assert selector_result == 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    query = 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    selector_result = wiki_synonym_preprocessor(query)
    assert selector_result == 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'


def test_wordnet_synonym_selector(wordnet_synonym_preprocessor):
    alcohol_query = 'MATCH (e:exposure {name: "alcohol"})-[:linked_to]->(d:disease) RETURN d.name'
    selector_result = wordnet_synonym_preprocessor(alcohol_query)
    assert selector_result == 'MATCH (e:exposure {name: "alcohol"})-[:linked_to]->(d:disease) RETURN d.name'
