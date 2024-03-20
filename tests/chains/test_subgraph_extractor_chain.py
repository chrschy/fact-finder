from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

from fact_finder.chains.subgraph_extractor_chain import SubgraphExtractorChain
from fact_finder.tools.subgraph_extension import SubgraphExpansion
from fact_finder.utils import load_chat_model

load_dotenv()


def test_subgraph_extraction_chain(llm, graph, cypher_query_preprocessors_chain_result, expected_graph_result):
    chain = SubgraphExtractorChain(llm, graph, subgraph_expansion=SubgraphExpansion(graph), use_subgraph_expansion=True)
    result = chain(cypher_query_preprocessors_chain_result)
    assert result[chain.output_key][chain.output_key] == expected_graph_result
    assert expected_graph_result[0] in result[chain.output_key]["expanded_nodes"]
    assert len(result[chain.output_key]["expanded_nodes"]) > len(result[chain.output_key][chain.output_key])
    assert chain.output_key in result.keys()


@pytest.fixture
def llm():
    # todo mock this
    return load_chat_model()


@pytest.fixture
def graph(expected_graph_result):
    graph = MagicMock(spec=Neo4jGraph)
    graph.query = MagicMock()
    graph.query.return_value = expected_graph_result
    return graph


@pytest.fixture
def cypher_query_preprocessors_chain_result():
    return {
        "cypher_query": "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name",
        "intermediate_steps": [
            {"question": "Which drugs are associated with epilepsy?"},
            {
                "FormatPreprocessor": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
                "LowerCasePropertiesCypherQueryPreprocessor": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
            },
        ],
        "preprocessed_cypher_query": 'MATCH (d:drug)-[:indication]->(dis:disease)\nWHERE dis.name = "epilepsy"\nRETURN d.name',
    }


@pytest.fixture
def expected_graph_result():
    return [
        {
            "d": {"name": "phenytoin", "index": 14141, "id": "dr:bayphth0000509", "source": "drugbank"},
            "r": (
                {"name": "phenytoin", "index": 14141, "id": "dr:bayphth0000509", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 84209, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 84209, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "phenytoin", "index": 14141, "id": "dr:bayphth0000509", "source": "drugbank"},
            "r": (
                {"name": "phenytoin", "index": 14141, "id": "dr:bayphth0000509", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "phenytoin", "index": 14141, "id": "dr:bayphth0000509", "source": "drugbank"},
            "r": (
                {"name": "phenytoin", "index": 14141, "id": "dr:bayphth0000509", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "valproic acid", "index": 14153, "id": "dr:bayphth0000301", "source": "drugbank"},
            "r": (
                {"name": "valproic acid", "index": 14153, "id": "dr:bayphth0000301", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "lamotrigine", "index": 14192, "id": "dr:bayphth0000351", "source": "drugbank"},
            "r": (
                {"name": "lamotrigine", "index": 14192, "id": "dr:bayphth0000351", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "lamotrigine", "index": 14192, "id": "dr:bayphth0000351", "source": "drugbank"},
            "r": (
                {"name": "lamotrigine", "index": 14192, "id": "dr:bayphth0000351", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "diazepam", "index": 14245, "id": "dr:bayphth0000290", "source": "drugbank"},
            "r": (
                {"name": "diazepam", "index": 14245, "id": "dr:bayphth0000290", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "clonazepam", "index": 14294, "id": "dr:bayphth0000277", "source": "drugbank"},
            "r": (
                {"name": "clonazepam", "index": 14294, "id": "dr:bayphth0000277", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "fosphenytoin", "index": 14330, "id": "dr:bayphth0002087", "source": "drugbank"},
            "r": (
                {"name": "fosphenytoin", "index": 14330, "id": "dr:bayphth0002087", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "mephenytoin", "index": 14505, "id": "dr:bayphth0000372", "source": "drugbank"},
            "r": (
                {"name": "mephenytoin", "index": 14505, "id": "dr:bayphth0000372", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 84209, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 84209, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "mephenytoin", "index": 14505, "id": "dr:bayphth0000372", "source": "drugbank"},
            "r": (
                {"name": "mephenytoin", "index": 14505, "id": "dr:bayphth0000372", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "neocitrullamon", "index": 14522, "id": "db13396", "source": "drugbank"},
            "r": (
                {"name": "neocitrullamon", "index": 14522, "id": "db13396", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "carbamazepine", "index": 14956, "id": "dr:bayphth0000249", "source": "drugbank"},
            "r": (
                {"name": "carbamazepine", "index": 14956, "id": "dr:bayphth0000249", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "carbamazepine", "index": 14956, "id": "dr:bayphth0000249", "source": "drugbank"},
            "r": (
                {"name": "carbamazepine", "index": 14956, "id": "dr:bayphth0000249", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "phenobarbital", "index": 14993, "id": "dr:bayphth0000503", "source": "drugbank"},
            "r": (
                {"name": "phenobarbital", "index": 14993, "id": "dr:bayphth0000503", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "phenobarbital", "index": 14993, "id": "dr:bayphth0000503", "source": "drugbank"},
            "r": (
                {"name": "phenobarbital", "index": 14993, "id": "dr:bayphth0000503", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "secobarbital", "index": 15297, "id": "dr:bayphth0000427", "source": "drugbank"},
            "r": (
                {"name": "secobarbital", "index": 15297, "id": "dr:bayphth0000427", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "primidone", "index": 15311, "id": "dr:bayphth0000471", "source": "drugbank"},
            "r": (
                {"name": "primidone", "index": 15311, "id": "dr:bayphth0000471", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "primidone", "index": 15311, "id": "dr:bayphth0000471", "source": "drugbank"},
            "r": (
                {"name": "primidone", "index": 15311, "id": "dr:bayphth0000471", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "lorazepam", "index": 15416, "id": "dr:bayphth0000362", "source": "drugbank"},
            "r": (
                {"name": "lorazepam", "index": 15416, "id": "dr:bayphth0000362", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 39815, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "pentobarbital", "index": 15430, "id": "dr:bayphth0000496", "source": "drugbank"},
            "r": (
                {"name": "pentobarbital", "index": 15430, "id": "dr:bayphth0000496", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "trimethadione", "index": 15434, "id": "dr:bayphth0000389", "source": "drugbank"},
            "r": (
                {"name": "trimethadione", "index": 15434, "id": "dr:bayphth0000389", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "paramethadione", "index": 15453, "id": "dr:bayphth0000492", "source": "drugbank"},
            "r": (
                {"name": "paramethadione", "index": 15453, "id": "dr:bayphth0000492", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "felbamate", "index": 15475, "id": "dr:bayphth0000533", "source": "drugbank"},
            "r": (
                {"name": "felbamate", "index": 15475, "id": "dr:bayphth0000533", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "thiopental", "index": 15834, "id": "dr:bayphth0000710", "source": "drugbank"},
            "r": (
                {"name": "thiopental", "index": 15834, "id": "dr:bayphth0000710", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "methylphenobarbital", "index": 15835, "id": "dr:bayphth0000696", "source": "drugbank"},
            "r": (
                {"name": "methylphenobarbital", "index": 15835, "id": "dr:bayphth0000696", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "mesuximide", "index": 15837, "id": "dr:bayphth0000401", "source": "drugbank"},
            "r": (
                {"name": "mesuximide", "index": 15837, "id": "dr:bayphth0000401", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "phenacemide", "index": 16475, "id": "dr:bayphth0000501", "source": "drugbank"},
            "r": (
                {"name": "phenacemide", "index": 16475, "id": "dr:bayphth0000501", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "metharbital", "index": 17065, "id": "dr:bayphth0001377", "source": "drugbank"},
            "r": (
                {"name": "metharbital", "index": 17065, "id": "dr:bayphth0001377", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "amobarbital", "index": 17066, "id": "dr:bayphth0000552", "source": "drugbank"},
            "r": (
                {"name": "amobarbital", "index": 17066, "id": "dr:bayphth0000552", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "ethadione", "index": 20341, "id": "db13799", "source": "drugbank"},
            "r": (
                {"name": "ethadione", "index": 20341, "id": "db13799", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
        {
            "d": {"name": "phensuximide", "index": 20354, "id": "dr:bayphth0000505", "source": "drugbank"},
            "r": (
                {"name": "phensuximide", "index": 20354, "id": "dr:bayphth0000505", "source": "drugbank"},
                "indication",
                {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
            ),
            "dis": {"name": "epilepsy", "index": 35641, "id": "sd:bayphth009438", "source": "mondo"},
        },
    ]
