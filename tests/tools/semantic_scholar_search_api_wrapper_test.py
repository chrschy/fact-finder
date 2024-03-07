import pytest

from fact_finder.tools.semantic_scholar_search_api_wrapper import SemanticScholarSearchApiWrapper


@pytest.fixture
def semantic_scholar_search_api_wrapper():
    return SemanticScholarSearchApiWrapper()

def test_search_by_question(semantic_scholar_search_api_wrapper):
    semantic_scholar_search_api_wrapper.search(
        #query="What are the symptoms of psiorasis?"
        keywords="psoriasis, symptoms"
    )