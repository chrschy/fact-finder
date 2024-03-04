from typing import Dict, List
from unittest.mock import MagicMock

import pytest
from langchain_community.graphs import Neo4jGraph

from fact_finder.tools.cypher_preprocessors.synonym_cypher_query_preprocessor import SynonymCypherQueryPreprocessor
from fact_finder.tools.synonym_finder.synonym_finder import SynonymFinder


_synonyms_with_match = ["ethanol", "alcohols", "alcohol by volume", "heat", "fever"]
_synonyms_without_match = ["alcoholic_beverage", "alcoholic_drink", "inebriant", "intoxicant"]


@pytest.fixture(scope="session")
def graph() -> Neo4jGraph:

    def get_graph_nodes(query: str) -> List[Dict[str, Dict[str, str]]]:
        if "exposure" in query:
            return [{"n": {"name": "ethanol", "id": "1"}}]
        if "disease" in query:
            return [{"n": {"name": "fever", "id": "2"}}]
        return []

    graph = MagicMock(spec=Neo4jGraph)
    graph.query = MagicMock()
    graph.query.side_effect = get_graph_nodes
    return graph


@pytest.fixture()
def synonyms(request) -> List[str]:
    if hasattr(request, "param") and request.param:
        return request.param
    return []


@pytest.fixture()
def synonym_preprocessor(synonyms: str, graph: Neo4jGraph):
    synonym_finder = MagicMock(spec=SynonymFinder)
    synonym_finder = MagicMock()
    synonym_finder.return_value = synonyms
    return SynonymCypherQueryPreprocessor(
        graph=graph, synonym_finder=synonym_finder, node_types=["exposure", "disease"]
    )


@pytest.mark.parametrize("synonyms", (_synonyms_with_match,), indirect=True)
def test_replaces_match_that_exists_in_graph(synonym_preprocessor):
    query = 'MATCH (e:exposure {name: "alcohol"})-[:linked_to]->(d:disease) RETURN d.name'
    selector_result = synonym_preprocessor(query)
    expected = 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    assert selector_result == expected


@pytest.mark.parametrize("synonyms", (_synonyms_with_match,), indirect=True)
def test_no_change_if_synonym_from_graph_is_already_used(synonym_preprocessor):
    query = 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    selector_result = synonym_preprocessor(query)
    assert selector_result == query


@pytest.mark.parametrize("synonyms", (_synonyms_without_match,), indirect=True)
def test_no_change_if_no_match_was_found(synonym_preprocessor):
    query = 'MATCH (e:exposure {name: "alcohol"})-[:linked_to]->(d:disease) RETURN d.name'
    selector_result = synonym_preprocessor(query)
    assert selector_result == query


@pytest.mark.parametrize("synonyms", (_synonyms_with_match,), indirect=True)
def test_replaces_multiple_matches_that_exists_in_graph(synonym_preprocessor):
    query = 'MATCH (e:exposure {name: "alcohol"})-[r]->(d:disease {name: "heat"}) RETURN r.name'
    selector_result = synonym_preprocessor(query)
    expected = 'MATCH (e:exposure {name: "ethanol"})-[r]->(d:disease {name: "fever"}) RETURN r.name'
    assert selector_result == expected


@pytest.mark.parametrize("synonyms", (_synonyms_with_match,), indirect=True)
def test_replaces_match_in_where_clause(synonym_preprocessor):
    query = 'MATCH (e:exposure)-[:linked_to]->(d:disease)\nWHERE d.name = "heat"\nRETURN e.name'
    selector_result = synonym_preprocessor(query)
    expected = 'MATCH (e:exposure)-[:linked_to]->(d:disease)\nWHERE d.name = "fever"\nRETURN e.name'
    assert selector_result == expected


@pytest.mark.parametrize("synonyms", (_synonyms_with_match,), indirect=True)
def test_replaces_match_in_where_clause_with_clutter(synonym_preprocessor):
    query = 'MATCH (d:disease)-[:linked_to]->(dr:drug)\nWHERE dr.name = "lollipop" and d.name = "heat" bla bla\nRETURN e.name'
    selector_result = synonym_preprocessor(query)
    expected = 'MATCH (d:disease)-[:linked_to]->(dr:drug)\nWHERE dr.name = "lollipop" and d.name = "fever" bla bla\nRETURN e.name'
    assert selector_result == expected


def test_does_not_change_cypher_query_if_no_matches_found(graph: Neo4jGraph):
    preproc = SynonymCypherQueryPreprocessor(graph=graph, synonym_finder=lambda x: [], node_types=[r"[^\s\"'{]+"])
    query = "MATCH (e:exposure)-[:linked_to]->(d:disease) RETURN e.name, d.name"
    selector_result = preproc(query)
    assert selector_result == query
