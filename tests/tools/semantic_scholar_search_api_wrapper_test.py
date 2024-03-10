import pytest

from fact_finder.tools.semantic_scholar_search_api_wrapper import SemanticScholarSearchApiWrapper


@pytest.fixture
def semantic_scholar_search_api_wrapper():
    return SemanticScholarSearchApiWrapper()


def test_search_by_abstract(semantic_scholar_search_api_wrapper):
    result = semantic_scholar_search_api_wrapper.search_by_abstracts(keywords="psoriasis, symptoms")
    assert 5 == len(result)
    assert result[0].startswith("Improve psoriasis")
    assert result[4].startswith("Prevalance and Odds")



def test_search_by_paper_content(semantic_scholar_search_api_wrapper):
    semantic_scholar_search_api_wrapper.search_by_paper_content(keywords="psoriasis, symptoms")
