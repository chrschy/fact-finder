from fact_finder.chains.cypher_preprocessors.lower_case_properties_cypher_query_preprocessor import (
    LowerCasePropertiesCypherQueryPreprocessor,
)


def test_producing_lower_case_for_given_property():
    preproc = LowerCasePropertiesCypherQueryPreprocessor(property_names=["name"])
    query1 = 'MATCH (e:exposure {name: "Ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    query2 = 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    processed_query = preproc(query1)
    assert processed_query == query2


def test_producing_lower_case_with_any_property():
    preproc = LowerCasePropertiesCypherQueryPreprocessor()
    query1 = 'MATCH (e:exposure {any_property: "Ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    query2 = 'MATCH (e:exposure {any_property: "ethanol"})-[:linked_to]->(d:disease) RETURN d.name'
    processed_query = preproc(query1)
    assert processed_query == query2


def test_producing_multiple_lower_case_with_any_property():
    preproc = LowerCasePropertiesCypherQueryPreprocessor()
    query1 = 'MATCH (e:exposure {any_property: "Ethanol"})-[:linked_to]->(d:disease {another_property: "HickUp"}) RETURN d.name'
    query2 = 'MATCH (e:exposure {any_property: "ethanol"})-[:linked_to]->(d:disease {another_property: "hickup"}) RETURN d.name'
    processed_query = preproc(query1)
    assert processed_query == query2


def test_producing_multiple_lower_case_for_multiple_given_properties():
    preproc = LowerCasePropertiesCypherQueryPreprocessor(property_names=["name", "disease_name"])
    query1 = 'MATCH (e:exposure {name: "Ethanol"})-[:linked_to]->(d:disease {disease_name: "HickUp"}) RETURN d.name'
    query2 = 'MATCH (e:exposure {name: "ethanol"})-[:linked_to]->(d:disease {disease_name: "hickup"}) RETURN d.name'
    processed_query = preproc(query1)
    assert processed_query == query2


def test_producing_lower_case_with_spaces_present():
    preproc = LowerCasePropertiesCypherQueryPreprocessor()
    query1 = 'MATCH (e:exposure)-[:linked_to]->(d:disease {disease_name: "Hick Up"}) RETURN d.name'
    query2 = 'MATCH (e:exposure)-[:linked_to]->(d:disease {disease_name: "hick up"}) RETURN d.name'
    processed_query = preproc(query1)
    assert processed_query == query2


def test_producing_lower_case_with_special_characters_present():
    preproc = LowerCasePropertiesCypherQueryPreprocessor()
    query1 = 'MATCH (e:exposure)-[:linked_to]->(d:disease {disease_name: "Hick-_!?=/\\+#Up"}) RETURN d.name'
    query2 = 'MATCH (e:exposure)-[:linked_to]->(d:disease {disease_name: "hick-_!?=/\\+#up"}) RETURN d.name'
    processed_query = preproc(query1)
    assert processed_query == query2


def test_producing_lower_case_for_assignment_in_where_clause():
    preproc = LowerCasePropertiesCypherQueryPreprocessor()
    query1 = 'MATCH (d:disease)-[:linked_to]-(e:exposure) WHERE e.name = "Ethanol" RETURN d.name'
    query2 = 'MATCH (d:disease)-[:linked_to]-(e:exposure) WHERE e.name = "ethanol" RETURN d.name'
    processed_query = preproc(query1)
    assert processed_query == query2


def test_producing_lower_case_for_multiple_assignments_in_where_clause():
    preproc = LowerCasePropertiesCypherQueryPreprocessor()
    query1 = 'MATCH (d:disease)-[:linked_to]-(e:exposure) WHERE e.name = "Ethanol" AND d.name = "HickUp" RETURN d.name'
    query2 = 'MATCH (d:disease)-[:linked_to]-(e:exposure) WHERE e.name = "ethanol" AND d.name = "hickup" RETURN d.name'
    processed_query = preproc(query1)
    assert processed_query == query2
