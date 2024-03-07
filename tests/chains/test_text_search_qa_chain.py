import pytest
from langchain_core.prompts import PromptTemplate

from fact_finder.chains.rag.text_search_qa_chain import TextSearchQAChain
from fact_finder.tools.semantic_scholar_search_api_wrapper import SemanticScholarSearchApiWrapper
from fact_finder.utils import load_chat_model


@pytest.fixture
def keyword_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. You get a user question in natural language. Please transform it into keywords that can be used in semantic scholar keyword search: Question: {question}",
    )


@pytest.fixture
def rag_answer_generation_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. You get a user question in natural language. Given the following context, please answer the given question based only on the context. Do not hallucinate. If you cannot answer based on the context, say 'Dunno'. Context: {context}, Question: {question}",
    )


@pytest.fixture
def text_search_qa_chain(keyword_prompt_template, rag_answer_generation_prompt_template) -> TextSearchQAChain:
    return TextSearchQAChain(
        semantic_scholar_search=SemanticScholarSearchApiWrapper(),
        llm=load_chat_model(),
        keyword_prompt_template=keyword_prompt_template,
        rag_answer_generation_template=rag_answer_generation_prompt_template,
    )


def test_simple_question(text_search_qa_chain):
    answer = text_search_qa_chain({"question": "Alternative causes of fever in malaria infections?"})
    assert ("Alternative causes of fever in malaria infections could include dengue, scrub typhus, leptospirosis, rickettsioses, AIDS, tuberculosis, "
            "respiratory and diarrhoeal infections, typhoid fever, infections caused by Tropheryma whipplei, Borrelia, Rickettsia felis, filarioses, "
            "and trypanosomiasis.") == answer["rag_output"]
