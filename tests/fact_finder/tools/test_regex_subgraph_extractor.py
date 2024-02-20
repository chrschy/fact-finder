from fact_finder.tools.sub_graph_extractor import RegexSubGraphExtractor


def test_extract_subgraph_preprocessor():
    extract_subgraph_preprocessor = RegexSubGraphExtractor()
    cypher_query = "MATCH (d:disease {name: 'schizophrenia'})-[:indication]->(g:drug) RETURN g"
    edited_cypher_query = extract_subgraph_preprocessor(cypher_query)
    expected_cypher_query = "MATCH (d:disease {name: 'schizophrenia'})-[l:indication]->(g:drug) RETURN d,l,g"
    assert isinstance(edited_cypher_query, str)
    assert len(cypher_query) + 5 == len(edited_cypher_query)
    assert len(expected_cypher_query) == len(edited_cypher_query)
