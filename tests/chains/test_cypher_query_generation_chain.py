from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration

from fact_finder.chains.custom_llm_chain import CustomLLMChain
from fact_finder.chains.cypher_query_generation_chain import CypherQueryGenerationChain
from fact_finder.prompt_templates import CYPHER_GENERATION_PROMPT
from fact_finder.utils import build_neo4j_graph, load_chat_model

load_dotenv()


class MockedCustomLLMChain(CustomLLMChain):
    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        return LLMResult(
            generations=[
                [
                    ChatGeneration(
                        text="MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name",
                        generation_info={"finish_reason": "stop", "logprobs": None},
                        message=AIMessage(
                            content="MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name"
                        ),
                    )
                ]
            ],
            llm_output={
                "model_name": "gpt-4",
                "token_usage": {"completion_tokens": 27, "prompt_tokens": 1162, "total_tokens": 1189},
            },
        )


def test_cypher_query_generation_chain_construct_predicate_descriptions(cypher_chain):
    how_many = 3
    predicate_descriptions = cypher_chain._construct_predicate_descriptions(how_many)
    assert isinstance(predicate_descriptions, str)
    predicate_descriptions_as_list = predicate_descriptions.split("\n")
    assert predicate_descriptions_as_list[0] == "Here are some descriptions to the most common relationships:"
    assert len(predicate_descriptions_as_list) - 1 == how_many

    how_many = 0
    predicate_descriptions = cypher_chain._construct_predicate_descriptions(how_many)
    assert isinstance(predicate_descriptions, str)
    assert predicate_descriptions == ""


@pytest.fixture()
def cypher_chain(llm_mocked, graph_mocked):
    chain = CypherQueryGenerationChain(llm=llm_mocked, graph=graph_mocked, cypher_prompt=CYPHER_GENERATION_PROMPT)
    return chain


@pytest.fixture()
def llm_mocked():
    return MagicMock(spec=BaseChatModel)


@pytest.fixture()
def graph_mocked():
    graph = MagicMock(spec=Neo4jGraph)
    return graph


def test_mocked(
    llm_mocked, graph_mocked, custom_chain_mocked, structured_schema, question, expected_filled_prompt_template
):
    with patch(
        "langchain_community.graphs.Neo4jGraph.get_structured_schema", new_callable=PropertyMock
    ) as mock_get_structured_schema:
        mock_get_structured_schema.return_value = structured_schema
        graph = graph_mocked
        chain = CypherQueryGenerationChain(llm=llm_mocked, graph=graph, cypher_prompt=CYPHER_GENERATION_PROMPT)
        chain.cypher_generation_chain = custom_chain_mocked
        result = chain(question)

        assert chain.output_key in result.keys()
        assert result["intermediate_steps"][0]["question"] == question
        assert (
            result["intermediate_steps"][1]["cypher_query_generation_filled_prompt_template"]
            == expected_filled_prompt_template
        )
        assert result["question"] == question
        assert (
            result["cypher_query"]
            == "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name"
        )
        assert len(result) == 3


@pytest.fixture()
def custom_chain_mocked(llm_mocked, expected_filled_prompt_template):
    custom_chain_mocked = MockedCustomLLMChain(llm=llm_mocked, prompt=CYPHER_GENERATION_PROMPT)
    custom_chain_mocked.filled_prompt_template = expected_filled_prompt_template
    return custom_chain_mocked


@pytest.fixture()
def question():
    return "Which drugs are associated with epilepsy?"


@pytest.fixture
def structured_schema():
    return {
        "node_props": {
            "exposure": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
            "drug": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
            "molecular_function": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
            "gene_or_protein": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
            "cellular_component": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
            "effect_or_phenotype": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
            "disease": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
            "pathway": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
            "anatomy": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
            "biological_process": [
                {"property": "id", "type": "STRING"},
                {"property": "name", "type": "STRING"},
                {"property": "source", "type": "STRING"},
                {"property": "index", "type": "INTEGER"},
            ],
        },
        "rel_props": {},
        "relationships": [
            {"start": "exposure", "type": "interacts_with", "end": "gene_or_protein"},
            {"start": "exposure", "type": "interacts_with", "end": "biological_process"},
            {"start": "exposure", "type": "interacts_with", "end": "molecular_function"},
            {"start": "exposure", "type": "interacts_with", "end": "cellular_component"},
            {"start": "exposure", "type": "parent_child", "end": "exposure"},
            {"start": "exposure", "type": "linked_to", "end": "disease"},
            {"start": "drug", "type": "enzyme", "end": "gene_or_protein"},
            {"start": "drug", "type": "synergistic_interaction", "end": "drug"},
            {"start": "drug", "type": "contraindication", "end": "disease"},
            {"start": "drug", "type": "target", "end": "gene_or_protein"},
            {"start": "drug", "type": "carrier", "end": "gene_or_protein"},
            {"start": "drug", "type": "transporter", "end": "gene_or_protein"},
            {"start": "drug", "type": "indication", "end": "disease"},
            {"start": "drug", "type": "side_effect", "end": "effect_or_phenotype"},
            {"start": "drug", "type": "off_label_use", "end": "disease"},
            {"start": "molecular_function", "type": "interacts_with", "end": "gene_or_protein"},
            {"start": "molecular_function", "type": "interacts_with", "end": "exposure"},
            {"start": "molecular_function", "type": "parent_child", "end": "molecular_function"},
            {"start": "gene_or_protein", "type": "associated_with", "end": "disease"},
            {"start": "gene_or_protein", "type": "associated_with", "end": "effect_or_phenotype"},
            {"start": "gene_or_protein", "type": "ppi", "end": "gene_or_protein"},
            {"start": "gene_or_protein", "type": "target", "end": "drug"},
            {"start": "gene_or_protein", "type": "expression_present", "end": "anatomy"},
            {"start": "gene_or_protein", "type": "expression_absent", "end": "anatomy"},
            {"start": "gene_or_protein", "type": "interacts_with", "end": "biological_process"},
            {"start": "gene_or_protein", "type": "interacts_with", "end": "molecular_function"},
            {"start": "gene_or_protein", "type": "interacts_with", "end": "cellular_component"},
            {"start": "gene_or_protein", "type": "interacts_with", "end": "pathway"},
            {"start": "gene_or_protein", "type": "interacts_with", "end": "exposure"},
            {"start": "gene_or_protein", "type": "enzyme", "end": "drug"},
            {"start": "gene_or_protein", "type": "transporter", "end": "drug"},
            {"start": "gene_or_protein", "type": "carrier", "end": "drug"},
            {"start": "cellular_component", "type": "parent_child", "end": "cellular_component"},
            {"start": "cellular_component", "type": "interacts_with", "end": "gene_or_protein"},
            {"start": "cellular_component", "type": "interacts_with", "end": "exposure"},
            {"start": "effect_or_phenotype", "type": "phenotype_present", "end": "disease"},
            {"start": "effect_or_phenotype", "type": "parent_child", "end": "effect_or_phenotype"},
            {"start": "effect_or_phenotype", "type": "phenotype_absent", "end": "disease"},
            {"start": "effect_or_phenotype", "type": "associated_with", "end": "gene_or_protein"},
            {"start": "effect_or_phenotype", "type": "side_effect", "end": "drug"},
            {"start": "disease", "type": "associated_with", "end": "gene_or_protein"},
            {"start": "disease", "type": "phenotype_present", "end": "effect_or_phenotype"},
            {"start": "disease", "type": "phenotype_absent", "end": "effect_or_phenotype"},
            {"start": "disease", "type": "parent_child", "end": "disease"},
            {"start": "disease", "type": "linked_to", "end": "exposure"},
            {"start": "disease", "type": "indication", "end": "drug"},
            {"start": "disease", "type": "contraindication", "end": "drug"},
            {"start": "disease", "type": "off_label_use", "end": "drug"},
            {"start": "pathway", "type": "parent_child", "end": "pathway"},
            {"start": "pathway", "type": "interacts_with", "end": "gene_or_protein"},
            {"start": "anatomy", "type": "parent_child", "end": "anatomy"},
            {"start": "anatomy", "type": "expression_present", "end": "gene_or_protein"},
            {"start": "anatomy", "type": "expression_absent", "end": "gene_or_protein"},
            {"start": "biological_process", "type": "interacts_with", "end": "gene_or_protein"},
            {"start": "biological_process", "type": "interacts_with", "end": "exposure"},
            {"start": "biological_process", "type": "parent_child", "end": "biological_process"},
        ],
    }


@pytest.mark.skip
def test_e2e(llm_e2e, graph_e2e):
    chain = CypherQueryGenerationChain(llm=llm_e2e, graph=graph_e2e, cypher_prompt=CYPHER_GENERATION_PROMPT)
    result = chain("Which drugs are associated with epilepsy?")
    assert (
        result["cypher_query"]
        == "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name"
    )


@pytest.fixture(scope="session")
def llm_e2e():
    return load_chat_model()


@pytest.fixture(scope="session")
def graph_e2e():
    return build_neo4j_graph()


@pytest.fixture
def expected_filled_prompt_template():
    return """Task: Generate Cypher statement to query a graph database described in the following schema.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
If there is no sensible Cypher statement for the given question and schema, state so and prepend SCHEMA_ERROR to your answer.
Any variables that are returned by the query must have readable names.
Remove modifying adjectives from the entities queried to the graph.

Schema:
Node properties are the following:
exposure {id: STRING, name: STRING, source: STRING, index: INTEGER},drug {id: STRING, name: STRING, source: STRING, index: INTEGER},molecular_function {id: STRING, name: STRING, source: STRING, index: INTEGER},gene_or_protein {id: STRING, name: STRING, source: STRING, index: INTEGER},cellular_component {id: STRING, name: STRING, source: STRING, index: INTEGER},effect_or_phenotype {id: STRING, name: STRING, source: STRING, index: INTEGER},disease {id: STRING, name: STRING, source: STRING, index: INTEGER},pathway {id: STRING, name: STRING, source: STRING, index: INTEGER},anatomy {id: STRING, name: STRING, source: STRING, index: INTEGER},biological_process {id: STRING, name: STRING, source: STRING, index: INTEGER}
Relationship properties are the following:

The relationships are the following:
(:exposure)-[:interacts_with]->(:gene_or_protein),(:exposure)-[:interacts_with]->(:biological_process),(:exposure)-[:interacts_with]->(:molecular_function),(:exposure)-[:interacts_with]->(:cellular_component),(:exposure)-[:parent_child]->(:exposure),(:exposure)-[:linked_to]->(:disease),(:drug)-[:enzyme]->(:gene_or_protein),(:drug)-[:synergistic_interaction]->(:drug),(:drug)-[:contraindication]->(:disease),(:drug)-[:target]->(:gene_or_protein),(:drug)-[:carrier]->(:gene_or_protein),(:drug)-[:transporter]->(:gene_or_protein),(:drug)-[:indication]->(:disease),(:drug)-[:side_effect]->(:effect_or_phenotype),(:drug)-[:off_label_use]->(:disease),(:molecular_function)-[:interacts_with]->(:gene_or_protein),(:molecular_function)-[:interacts_with]->(:exposure),(:molecular_function)-[:parent_child]->(:molecular_function),(:gene_or_protein)-[:associated_with]->(:disease),(:gene_or_protein)-[:associated_with]->(:effect_or_phenotype),(:gene_or_protein)-[:expression_present]->(:anatomy),(:gene_or_protein)-[:ppi]->(:gene_or_protein),(:gene_or_protein)-[:interacts_with]->(:biological_process),(:gene_or_protein)-[:interacts_with]->(:cellular_component),(:gene_or_protein)-[:interacts_with]->(:molecular_function),(:gene_or_protein)-[:interacts_with]->(:pathway),(:gene_or_protein)-[:interacts_with]->(:exposure),(:gene_or_protein)-[:target]->(:drug),(:gene_or_protein)-[:carrier]->(:drug),(:gene_or_protein)-[:expression_absent]->(:anatomy),(:gene_or_protein)-[:enzyme]->(:drug),(:gene_or_protein)-[:transporter]->(:drug),(:cellular_component)-[:parent_child]->(:cellular_component),(:cellular_component)-[:interacts_with]->(:gene_or_protein),(:cellular_component)-[:interacts_with]->(:exposure),(:effect_or_phenotype)-[:phenotype_present]->(:disease),(:effect_or_phenotype)-[:parent_child]->(:effect_or_phenotype),(:effect_or_phenotype)-[:associated_with]->(:gene_or_protein),(:effect_or_phenotype)-[:phenotype_absent]->(:disease),(:effect_or_phenotype)-[:side_effect]->(:drug),(:disease)-[:associated_with]->(:gene_or_protein),(:disease)-[:phenotype_present]->(:effect_or_phenotype),(:disease)-[:phenotype_absent]->(:effect_or_phenotype),(:disease)-[:parent_child]->(:disease),(:disease)-[:contraindication]->(:drug),(:disease)-[:off_label_use]->(:drug),(:disease)-[:indication]->(:drug),(:disease)-[:linked_to]->(:exposure),(:pathway)-[:parent_child]->(:pathway),(:pathway)-[:interacts_with]->(:gene_or_protein),(:anatomy)-[:parent_child]->(:anatomy),(:anatomy)-[:expression_present]->(:gene_or_protein),(:anatomy)-[:expression_absent]->(:gene_or_protein),(:biological_process)-[:interacts_with]->(:gene_or_protein),(:biological_process)-[:interacts_with]->(:exposure),(:biological_process)-[:parent_child]->(:biological_process)



Note: 
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
Which drugs are associated with epilepsy?"""
