from fact_finder.tools.cypher_preprocessors.size_to_count_preprocessor import (
    SizeToCountPreprocessor,
)


def test_size_to_count():
    preproc = SizeToCountPreprocessor()
    cypher_query = "MATCH (n)\nWITH n, SIZE([(n)--()|1]) AS num_edges\nWHERE num_edges = 1\nRETURN n"
    processed_query = "MATCH (n)\nWITH n, COUNT{[(n)--()|1]} AS num_edges\nWHERE num_edges = 1\nRETURN n"
    assert preproc(cypher_query) == processed_query


def test_size_to_count_with_whitespace():
    preproc = SizeToCountPreprocessor()
    cypher_query = "MATCH (n)\nWITH n, SIZE ([(n)--()|1]) AS num_edges\nWHERE num_edges = 1\nRETURN n"
    processed_query = "MATCH (n)\nWITH n, COUNT{[(n)--()|1]} AS num_edges\nWHERE num_edges = 1\nRETURN n"
    assert preproc(cypher_query) == processed_query


def test_size_to_count_with_multiple_levels_of_inner_brackets():
    preproc = SizeToCountPreprocessor()
    cypher_query = "MATCH (n)\nWHERE SIZE ((())(foo(bar))) = 1\nRETURN n"
    processed_query = "MATCH (n)\nWHERE COUNT{(())(foo(bar))} = 1\nRETURN n"
    assert preproc(cypher_query) == processed_query
