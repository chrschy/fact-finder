from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import PromptTemplate

from fact_finder.chains.graph_summary_chain import GraphSummaryChain
from fact_finder.utils import load_chat_model


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


@pytest.fixture
def graph_summary_chain(graph_summary_template) -> GraphSummaryChain:
    return GraphSummaryChain(llm=load_chat_model(), graph_summary_template=graph_summary_template)


def test_simple_question(graph_summary_chain):
    answer = graph_summary_chain({"sub_graph": "(psoriasis, is a, disease)"})
    print(answer)
    assert answer["summary"].startswith("Psoriasis is a disease")
