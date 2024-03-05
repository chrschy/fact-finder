from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_community.graphs import Neo4jGraph
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration

from fact_finder.chains.cypher_query_generation_chain import CypherQueryGenerationChain
from fact_finder.prompt_templates import CYPHER_GENERATION_PROMPT
from fact_finder.utils import build_neo4j_graph, load_chat_model

load_dotenv()


@pytest.fixture
def llm_chain_result() -> LLMResult:
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


@pytest.fixture()
def chat_model() -> BaseChatModel:
    return MagicMock(spec=BaseChatModel)


@pytest.fixture
def llm_chain(chat_model, llm_chain_result) -> LLMChain:
    class LLMChainMock(LLMChain):
        def generate(
            self, input_list: List[Dict[str, Any]], run_manager: CallbackManagerForChainRun | None = None
        ) -> LLMResult:
            return llm_chain_result

    return LLMChainMock(llm=chat_model, prompt=CYPHER_GENERATION_PROMPT)


@pytest.fixture
def cypher_chain(llm_chain, graph):
    chain = CypherQueryGenerationChain(llm=llm_chain, graph=graph, prompt_template=CYPHER_GENERATION_PROMPT)
    return chain


@pytest.fixture
def graph() -> Neo4jGraph:
    graph = MagicMock(spec=Neo4jGraph)
    return graph


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


def test_mocked(chat_model: BaseChatModel, llm_chain: LLMChain, graph: Neo4jGraph, structured_schema, question):
    with patch(
        "langchain_community.graphs.Neo4jGraph.get_structured_schema", new_callable=PropertyMock
    ) as mock_get_structured_schema:
        mock_get_structured_schema.return_value = structured_schema
        graph = graph
        chain = CypherQueryGenerationChain(llm=chat_model, graph=graph, prompt_template=CYPHER_GENERATION_PROMPT)
        chain.cypher_generation_chain = llm_chain
        result = chain(question)

        assert chain.output_key in result.keys()
        assert result["intermediate_steps"][0]["question"] == question
        assert result["question"] == question
        assert (
            result["cypher_query"]
            == "MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d.name"
        )
        assert len(result) == 3


@pytest.fixture
def question() -> str:
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
    chain = CypherQueryGenerationChain(llm=llm_e2e, graph=graph_e2e, prompt_template=CYPHER_GENERATION_PROMPT)
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
