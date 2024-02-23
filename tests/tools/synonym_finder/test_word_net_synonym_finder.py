import pytest

from fact_finder.tools.synonym_finder.word_net_synonym_finder import WordNetSynonymFinder


@pytest.mark.parametrize("query", ("table", "alcohol"))
def test_query_is_returned_as_potential_synonym(query: str):
    finder = WordNetSynonymFinder()
    res = finder(query)
    assert query in res


def test_finds_synonyms_for_all_meanings():
    finder = WordNetSynonymFinder()
    result = finder("table")
    meanings = ["tabular_array", "postpone", "board"]
    assert all(m in result for m in meanings)
