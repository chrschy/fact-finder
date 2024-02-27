from fact_finder.qa_service.cypher_preprocessors.format_preprocessor import FormatPreprocessor


def test_on_empty_string():
    preprocessor = FormatPreprocessor()
    assert preprocessor("") == ""


def test_well_formed_query_remains_unchanged():
    query = (
        'MATCH (disease1:disease {name: "psoriasis"})-[:linked_to]->(exposure:exposure)-[:linked_to]->(disease2:disease {name: "scalp disease"})\n'
        'WHERE p.name = "phenotype"\n'
        "RETURN disease1, exposure, disease2"
    )
    preprocessor = FormatPreprocessor()
    assert preprocessor(query) == query


def test_newlines_get_added():
    query = (
        'MATCH (disease1:disease {name: "psoriasis"})-[:linked_to]->(exposure:exposure)-[:linked_to]->(disease2:disease {name: "scalp disease"})'
        'WHERE p.name = "phenotype"'
        "RETURN disease1, exposure, disease2"
    )
    formated_query = (
        'MATCH (disease1:disease {name: "psoriasis"})-[:linked_to]->(exposure:exposure)-[:linked_to]->(disease2:disease {name: "scalp disease"})\n'
        'WHERE p.name = "phenotype"\n'
        "RETURN disease1, exposure, disease2"
    )
    preprocessor = FormatPreprocessor()
    assert preprocessor(query) == formated_query


def test_spaces_in_node_get_removed():
    query = (
        'MATCH ( disease1 : disease  { name : "psoriasis" } )-[:linked_to]->( exposure : exposure )-[:linked_to]->(disease2:disease {name: "scalp disease"})\n'
        'WHERE p.name = "phenotype"\n'
        "RETURN disease1, exposure, disease2"
    )
    formated_query = (
        'MATCH (disease1:disease {name: "psoriasis"})-[:linked_to]->(exposure:exposure)-[:linked_to]->(disease2:disease {name: "scalp disease"})\n'
        'WHERE p.name = "phenotype"\n'
        "RETURN disease1, exposure, disease2"
    )
    preprocessor = FormatPreprocessor()
    assert preprocessor(query) == formated_query


def test_spaces_in_edges_get_removed():
    query = (
        'MATCH (disease1:disease {name: "psoriasis"}) - [ : linked_to ] - > (exposure:exposure) - [ : linked_to ] - > (disease2:disease {name: "scalp disease"})\n'
        'WHERE p.name = "phenotype"\n'
        "RETURN disease1, exposure, disease2"
    )
    formated_query = (
        'MATCH (disease1:disease {name: "psoriasis"})-[:linked_to]->(exposure:exposure)-[:linked_to]->(disease2:disease {name: "scalp disease"})\n'
        'WHERE p.name = "phenotype"\n'
        "RETURN disease1, exposure, disease2"
    )
    preprocessor = FormatPreprocessor()
    assert preprocessor(query) == formated_query


def test_single_quotes_to_double_quotes():
    query = (
        "MATCH (disease1:disease {name: 'psoriasi\\'s'})-[:linked_to]->(exposure:exposure)-[:linked_to]->(disease2:disease {name: 'scalp disease'})\n"
        "WHERE p.name = 'phenotype'\n"
        "RETURN disease1, exposure, disease2"
    )
    formated_query = (
        'MATCH (disease1:disease {name: "psoriasi\\\'s"})-[:linked_to]->(exposure:exposure)-[:linked_to]->(disease2:disease {name: "scalp disease"})\n'
        'WHERE p.name = "phenotype"\n'
        "RETURN disease1, exposure, disease2"
    )
    preprocessor = FormatPreprocessor()
    assert preprocessor(query) == formated_query


def test_match_formatings_also_work_in_exists_block():
    query = (
        'MATCH (d:disease {name: "psoriasis"})-[:indication]->(drug:drug)\n'
        "WHERE EXISTS( ( : disease { name :'psoriatic arthriti\\'s' } ) - [ : indication ] - > ( drug ) )"
        "RETURN drug.name"
    )
    formated_query = (
        'MATCH (d:disease {name: "psoriasis"})-[:indication]->(drug:drug)\n'
        'WHERE EXISTS((:disease {name: "psoriatic arthriti\\\'s"})-[:indication]->(drug))\n'
        "RETURN drug.name"
    )
    preprocessor = FormatPreprocessor()
    assert preprocessor(query) == formated_query


def test_single_quotes_in_double_quoted_string_get_escaped():
    query = (
        "MATCH (disease1:disease {name: \"pso'riasi's\"})-[:linked_to]->(exposure:exposure)-[:linked_to]->(disease2:disease {name: 'scalp disease'})\n"
        "WHERE p.name = \"phe'no'type\"\n"
        "RETURN disease1, exposure, disease2"
    )
    formated_query = (
        'MATCH (disease1:disease {name: "pso\\\'riasi\\\'s"})-[:linked_to]->(exposure:exposure)-[:linked_to]->(disease2:disease {name: "scalp disease"})\n'
        "WHERE p.name = \"phe\\'no\\'type\"\n"
        "RETURN disease1, exposure, disease2"
    )
    preprocessor = FormatPreprocessor()
    assert preprocessor(query) == formated_query
