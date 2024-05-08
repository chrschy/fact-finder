import pytest
from fact_finder.chains.graph_summary_chain import GraphSummaryChain
from langchain_core.prompts import PromptTemplate
from tests.chains.helpers import build_llm_mock


def test_simple_question(graph_summary_chain: GraphSummaryChain):
    answer = graph_summary_chain({"sub_graph": "(psoriasis, is a, disease)"})
    print(answer)
    assert answer["summary"].startswith("Psoriasis is a disease")


@pytest.fixture
def graph_summary_chain(graph_summary_template: PromptTemplate) -> GraphSummaryChain:
    return GraphSummaryChain(
        llm=build_llm_mock("Psoriasis is a disease."), graph_summary_template=graph_summary_template
    )


@pytest.fixture
def graph_summary_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["sub_graph"],
        template="""
Verbalize the given triplets of a subgraph to natural text. Use all triplets for the verbalization.
 
Triplets of the subgraph:
{sub_graph}
""",
    )
