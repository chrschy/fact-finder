import asyncio
import os
from enum import Enum
from typing import Dict, List, Optional

from fact_finder.chains.graph_qa_chain import GraphQAChainOutput
from fact_finder.ui.graph_conversion import Subgraph, convert_subgraph
from pydantic import BaseModel


class PipelineOptions(str, Enum):
    LLM = "LLM only"
    GRAPH = "Graph"
    DOC = "Document"
    GRAPH_DOC = "Graph+Document"
    GRAPH_SUM = "Graph Summary"
    EVAL = "Evaluation"


class PipelineResponse(BaseModel):
    status: str
    llm_answer: str
    graph_answer: str
    graph_query: str
    graph_response: list
    graph_prompt_cypher: str
    graph_prompt_answer: str
    graph_subgraph: Subgraph
    graph_subgraph_neo4j: List[dict]
    graph_summary: str
    graph_expanded_subgraph: Optional[Subgraph]
    graph_expanded_summary: Optional[str]
    rag_answer: str
    rag_keywords: str
    rag_paragraphs: str
    rag_prompt_answer: str
    graph_rag_answer: str
    subgraph_cypher: str


async def call_neo4j_rag(message: str, session_state):
    return session_state.neo4j_rag_chain.invoke(message)  # FIXME use ainvoke?


async def call_neo4j(message: str, session_state):
    return session_state.neo4j_chain.invoke(message)  # FIXME use ainvoke?


async def call_llm(message: str, session_state):
    return session_state.llm_chain.invoke(message)  # FIXME use ainvoke?


async def call_rag(message: str, session_state):
    return session_state.rag_chain.invoke(message)  # FIXME use ainvoke?


async def call_summary(sub_graph: str, session_state):
    return session_state.summary_chain.invoke(sub_graph)  # FIXME use ainvoke?


def call_chains(message: str, pipelines_selected: List[str], session_state):
    results = {}

    if PipelineOptions.LLM.value in pipelines_selected:
        results[PipelineOptions.LLM.value] = asyncio.run(call_llm(message, session_state))
    else:
        results[PipelineOptions.LLM.value] = {"text": ""}

    if PipelineOptions.GRAPH.value in pipelines_selected:
        results[PipelineOptions.GRAPH.value] = asyncio.run(call_neo4j(message, session_state))
    else:
        results[PipelineOptions.GRAPH.value] = {
            "graph_qa_output": GraphQAChainOutput(
                question="",
                cypher_query="",
                graph_response=[],
                answer="",
                evidence_sub_graph=[],
                expanded_evidence_subgraph=[],
            )
        }

    if PipelineOptions.GRAPH_DOC.value in pipelines_selected:
        results[PipelineOptions.GRAPH_DOC.value] = asyncio.run(call_neo4j_rag(message, session_state))
    else:
        results[PipelineOptions.GRAPH_DOC.value] = {
            "graph_qa_output": GraphQAChainOutput(
                question="",
                cypher_query="",
                graph_response=[],
                answer="",
                evidence_sub_graph=[],
                expanded_evidence_subgraph=[],
            )
        }

    if PipelineOptions.DOC.value in pipelines_selected:
        results[PipelineOptions.DOC.value] = asyncio.run(call_rag(message, session_state))
    else:
        results[PipelineOptions.DOC.value] = {"rag_output": ""}

    if "intermediate_steps" in results[PipelineOptions.GRAPH.value]:
        results[PipelineOptions.GRAPH.value]["intermediate_steps"] = convert_intermediate_steps(
            results[PipelineOptions.GRAPH.value]["intermediate_steps"]
        )

    if "intermediate_steps" in results[PipelineOptions.GRAPH_DOC.value]:
        results[PipelineOptions.GRAPH_DOC.value]["intermediate_steps"] = convert_intermediate_steps(
            results[PipelineOptions.GRAPH_DOC.value]["intermediate_steps"]
        )

    if "intermediate_steps" in results[PipelineOptions.DOC.value]:
        results[PipelineOptions.DOC.value]["intermediate_steps"] = convert_intermediate_steps(
            results[PipelineOptions.DOC.value]["intermediate_steps"]
        )

    # print(results)

    return results


def convert_intermediate_steps(intermediate_steps):
    intermediate_steps_converted = {}
    for step in intermediate_steps:
        if type(step) is tuple:
            intermediate_steps_converted[step[0]] = step[1]
        else:
            intermediate_steps_converted[list(step.keys())[0]] = list(step.values())[0]

    return intermediate_steps_converted


def request_pipeline(text_data: str, pipelines_selected: List[str], session_state) -> PipelineResponse:

    results = call_chains(text_data, pipelines_selected, session_state)
    graph_result: GraphQAChainOutput = results[PipelineOptions.GRAPH.value]["graph_qa_output"]
    graph_rag_result: GraphQAChainOutput = results[PipelineOptions.GRAPH_DOC.value]["graph_qa_output"]
    subgraph, triplets = convert_subgraph(graph_result.evidence_sub_graph, graph_result.graph_response)
    if len(graph_result.expanded_evidence_subgraph) > 0:
        expanded_subgraph, expanded_triples = convert_subgraph(
            graph_result.expanded_evidence_subgraph, graph_result.graph_response
        )
    else:
        expanded_subgraph, expanded_triples = None, None
    try:
        graph_prompt_answer = results[PipelineOptions.GRAPH.value]["intermediate_steps"][
            "AnswerGenerationChain_filled_prompt"
        ]
    except KeyError as e:
        print(e)
        graph_prompt_answer = ""
    return PipelineResponse(
        status="success",
        llm_answer=results[PipelineOptions.LLM.value]["text"],
        graph_answer=graph_result.answer,
        graph_query=graph_result.cypher_query,
        graph_response=graph_result.graph_response,
        graph_prompt_cypher=(
            results[PipelineOptions.GRAPH.value]["intermediate_steps"]["CypherQueryGenerationChain_filled_prompt"]
            if "intermediate_steps" in results[PipelineOptions.GRAPH.value]
            else ""
        ),
        graph_prompt_answer=(
            graph_prompt_answer if "intermediate_steps" in results[PipelineOptions.GRAPH.value] else ""
        ),
        graph_subgraph=subgraph,
        graph_subgraph_neo4j=graph_result.evidence_sub_graph,
        graph_summary=(
            asyncio.run(call_summary(triplets, session_state))["summary"]
            if PipelineOptions.GRAPH_SUM.value in pipelines_selected
            else ""
        ),
        graph_expanded_subgraph=expanded_subgraph,
        graph_expanded_summary=(
            asyncio.run(call_summary(expanded_triples, session_state))["summary"]
            if PipelineOptions.GRAPH_SUM.value in pipelines_selected
            else ""
        ),
        rag_answer=results[PipelineOptions.DOC.value]["rag_output"],
        rag_keywords=(
            results[PipelineOptions.DOC.value]["intermediate_steps"]["search_keywords"]
            if "intermediate_steps" in results[PipelineOptions.DOC.value]
            else ""
        ),
        rag_paragraphs=(
            results[PipelineOptions.DOC.value]["intermediate_steps"]["semantic_scholar_result"]
            if "intermediate_steps" in results[PipelineOptions.DOC.value]
            else ""
        ),
        rag_prompt_answer=(
            results[PipelineOptions.DOC.value]["intermediate_steps"]["TextSearchQAChain_filled_prompt"]
            if "intermediate_steps" in results[PipelineOptions.DOC.value]
            else ""
        ),
        graph_rag_answer=graph_rag_result.answer,
        subgraph_cypher=(
            results[PipelineOptions.GRAPH.value]["intermediate_steps"]["subgraph_cypher"]
            if "intermediate_steps" in results[PipelineOptions.GRAPH.value]
            else ""
        ),
    )
