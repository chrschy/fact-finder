import os
from typing import Dict, List, Any

from langchain.chains import LLMChain
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI


def concatenate_with_headers(answers: List[Dict[str, str]]) -> str:
    result = ""
    for answer in answers:
        for header, text in answer.items():
            result += header + "\n" + text + "\n\n"
    return result


def build_neo4j_graph() -> Neo4jGraph:
    """

    :rtype: object
    """
    NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PW = os.getenv("NEO4J_PW", "opensesame")
    return Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PW)


def get_model_from_env():
    return os.getenv("LLM", "gpt-4o")


def load_chat_model() -> BaseChatModel:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    assert OPENAI_API_KEY is not None, "An OpenAI API key has to be set as environment variable OPENAI_API_KEY."
    if os.getenv("AZURE_OPENAI_ENDPOINT") is not None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert endpoint is not None
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = "2023-05-15"
        os.environ["AZURE_OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
        return AzureChatOpenAI(openai_api_version=api_version, azure_deployment=deployment_name)
    model = get_model_from_env()
    return ChatOpenAI(model=model, streaming=False, temperature=0, api_key=OPENAI_API_KEY)


def graph_result_contains_triple(graph_result_entry):
    return len(get_triples_from_graph_result(graph_result_entry)) > 0


def get_triples_from_graph_result(graph_result_entry) -> List[dict]:
    return [value for key, value in graph_result_entry.items() if type(value) is tuple]


def fill_prompt_template(llm_chain: LLMChain, inputs: Dict[str, Any]) -> str:
    return llm_chain.prep_prompts([inputs])[0][0].text
