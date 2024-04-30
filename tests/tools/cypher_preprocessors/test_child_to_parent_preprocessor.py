from unittest.mock import MagicMock

import pytest
from fact_finder.tools.cypher_preprocessors.child_to_parent_preprocessor import (
    ChildToParentPreprocessor,
)
from langchain_community.graphs import Neo4jGraph


def test_replaces_property_name_in_node(preprocessor: ChildToParentPreprocessor):
    query1 = 'MATCH (e:exposure {name: "child1"})-[:linked_to]->(d:disease) RETURN d.name'
    query2 = 'MATCH (e:exposure {name: "parent1"})-[:linked_to]->(d:disease) RETURN d.name'
    processed_query = preprocessor(query1)
    assert processed_query == query2


def test_replaces_property_name_in_where_clause(preprocessor: ChildToParentPreprocessor):
    query1 = 'MATCH (d:disease)-[:linked_to]-(e:exposure) WHERE e.name = "child2" RETURN d.name'
    query2 = 'MATCH (d:disease)-[:linked_to]-(e:exposure) WHERE e.name = "parent2" RETURN d.name'
    processed_query = preprocessor(query1)
    assert processed_query == query2


def test_replaces_property_name_in_multi_element_list_in_where_clause(preprocessor: ChildToParentPreprocessor):
    query1 = 'MATCH (d:disease)-[:linked_to]-(e:exposure) WHERE e.name IN ["child3", "child1", "child1"] RETURN d'
    query2 = 'MATCH (d:disease)-[:linked_to]-(e:exposure) WHERE e.name IN ["parent3", "parent1", "parent1"] RETURN d'
    processed_query = preprocessor(query1)
    assert processed_query == query2


@pytest.fixture
def preprocessor() -> ChildToParentPreprocessor:
    graph = MagicMock(spec=Neo4jGraph)
    graph.query = MagicMock()
    graph.query.return_value = [
        {"child.name": "child1", "parent.name": "parent1"},
        {"child.name": "child2", "parent.name": "parent2"},
        {"child.name": "child3", "parent.name": "parent3"},
    ]
    return ChildToParentPreprocessor(graph, "parent_child", name_property="name")
