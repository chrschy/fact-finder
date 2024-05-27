from fact_finder.tools.cypher_preprocessors.always_distinct_preprocessor import (
    AlwaysDistinctCypherQueryPreprocessor,
)


def test_adds_distinct_keyword():
    preproc = AlwaysDistinctCypherQueryPreprocessor()
    query1 = 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    query2 = 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN DISTINCT d.name'
    processed_query = preproc(query1)
    assert processed_query == query2


def test_does_nothing_if_distinct_keyword_already_present():
    preproc = AlwaysDistinctCypherQueryPreprocessor()
    query1 = 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN DISTINCT d.name'
    processed_query = preproc(query1)
    assert processed_query == query1
