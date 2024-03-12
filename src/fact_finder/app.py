import asyncio
import sys
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from PIL import Image
from pyvis.network import Network

import fact_finder.config.primekg_config as graph_config
import fact_finder.config.simple_config as llm_config
from fact_finder.qa_service.neo4j_langchain_qa_service import Subgraph
from fact_finder.chains.graph_qa_chain import GraphQAChainOutput
from fact_finder.utils import load_chat_model

load_dotenv()


############################################################
## Constants
############################################################
PLACEHOLDER_QUERY = "Please insert your query"

EXAMPLE_1 = "What are the phenotypes associated with cardioacrofacial dysplasia?"
EXAMPLE_2 = "What are the genes responsible for psoriasis?"
EXAMPLE_3 = "Which diseases involve PINK1?"
EXAMPLE_4 = "Which drugs could be promising candidates to treat Parkinson?"

max_width_str = f"max-width: 65%;"
style = "<style>mark.entity { display: inline-block }</style>"
graph_options = """ 
var options = {
  "edges": {
    "arrows": {
      "to": {
        "enabled": true,
        "scaleFactor": 1.2
      }
    }
  },
  "interaction": {
    "hover": true
  }
}
"""

COLORS = {"edge": "#aaa", "default": "#eee", "query": "#ffbe00", "answer": "#46d000"}

TYPE_MAPPING = {"d": "disease", "g": "gene/protein", "p": "phenotypes "}


############################################################
## Page configuration
############################################################
st.set_page_config(
    page_title="Fact Finder",
    menu_items={"Get Help": None, "Report a bug": None, "About": "## Demonstrator Fact Finder"},
    #  layout="wide")
)

st.markdown(
    f"""
    <style>
    .appview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
    unsafe_allow_html=True,
)

# Header
col1, col2 = st.columns([0.4, 1.4])

with col1:
    image = Image.open("src/img/logo.png")
    st.image(image, width=300)
with col2:
    st.header(" ")
    st.header("Fact Finder", divider="grey")


############################################################
## initialize pipeline
############################################################
if "counter" not in st.session_state:
    st.session_state.counter = 0
    with st.spinner("Initializing Chains..."):
        chat_model = load_chat_model()
        st.session_state.neo4j_chain = graph_config.build_chain(chat_model, sys.argv)
        st.session_state.llm_chain = llm_config.build_chain(chat_model, sys.argv)


############################################################
## Function definitions
############################################################


def get_html(html: str, legend=False):
    """Convert HTML so it can be rendered."""
    WRAPPER = (
        """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1.5rem">{}</div>"""
    )
    if legend:
        WRAPPER = """<div style="overflow-x: auto; padding: 1rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


async def call_neo4j(message):
    return st.session_state.neo4j_chain.invoke(message)  # FIXME use ainvoke?


async def call_llm(message):
    return st.session_state.llm_chain.invoke(message)  # FIXME use ainvoke?


async def call_chains(message):
    results = await asyncio.gather(call_neo4j(message), call_llm(message))
    print(results)
    return results


def convert_subgraph(graph: List[Dict[str, Any]], result: List[Dict[str, Any]]) -> Subgraph:
    graph_converted = Subgraph(nodes=[], edges=[])

    try:
        result_ents = []
        for res in result:
            result_ents += res.values()

        idx_rel = 0
        for triplet in graph:
            trip = [value for key, value in triplet.items() if type(value) is tuple][0]
            head_type = [key for key, value in triplet.items() if value == trip[0]]
            tail_type = [key for key, value in triplet.items() if value == trip[2]]
            head_type = head_type[0] if len(head_type) > 0 else ""
            tail_type = tail_type[0] if len(tail_type) > 0 else ""
            node_head = trip[0] if "index" in trip[0] else list(triplet.values())[0]
            node_tail = trip[2] if "index" in trip[2] else list(triplet.values())[2]

            if "index" in node_head and node_head["index"] not in [node.id for node in graph_converted.nodes]:
                graph_converted.nodes.append(
                    Node(
                        id=node_head["index"],
                        type=head_type,
                        name=node_head["name"],
                        in_query=False,
                        in_answer=node_head["name"] in result_ents,
                    )
                )
            if "index" in node_tail and node_tail["index"] not in [node.id for node in graph_converted.nodes]:
                graph_converted.nodes.append(
                    Node(
                        id=node_tail["index"],
                        type=tail_type,
                        name=node_tail["name"],
                        in_query=False,
                        in_answer=node_tail["name"] in result_ents,
                    )
                )
            if "index" in node_head and "index" in node_tail:
                graph_converted.edges.append(
                    Edge(
                        id=idx_rel,
                        type=trip[1],
                        name=trip[1],
                        source=node_head["index"],
                        target=node_tail["index"],
                        in_query=False,
                        in_answer=node_tail["name"] in result_ents,
                    )
                )
                idx_rel += 1

    except Exception as e:
        print(e)

    return graph_converted


def request_pipeline(text_data: str):
    # try:
    results = asyncio.run(call_chains(text_data))
    graph_result: GraphQAChainOutput = results[0]["graph_qa_output"]
    return {
        "status": "success",
        "query": graph_result.cypher_query,
        "response": graph_result.graph_response,
        "answer_graph": graph_result.answer,
        "answer_llm": results[1]["text"],
        "graph": results[0]["sub_graph"] if "sub_graph" in results[0] else [],
        "graph_neo4j": results[0]["sub_graph_neo4j"] if "sub_graph_neo4j" in results[0] else [],
        "graph_summary": results[0]["sub_graph_summary"] if "sub_graph_summary" in results[0] else "",
        # "graph": convert_subgraph(graph_result.evidence_sub_graph, graph_result.graph_response),
        # "graph_neo4j": graph_result.evidence_sub_graph,
    }
    # except Exception as e:
    #     print(e)
    #     return {
    #         "status": "error",
    #         "query": "",
    #         "response": "",
    #         "answer_graph": "",
    #         "answer_llm": "",
    #         "graph": {},
    #         "graph_neo4j": [],
    #     }


def generate_graph(graph: Subgraph, send_request=False):
    net = Network(
        height="530px", width="100%"
    )  # , bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)

    for node in graph.nodes:
        color = COLORS["query"] if node.in_query else COLORS["answer"] if node.in_answer else COLORS["default"]
        net.add_node(node.id, label=node.name, title=node.type, mass=1, shape="ellipse", color=color)

    for edge in graph.edges:
        color = COLORS["answer"] if edge.in_answer else COLORS["edge"]
        net.add_edge(edge.source, edge.target, width=1, title=edge.name, arrowStrikethrough=False, color=color)

    net.force_atlas_2based()  # barnes_hut()  force_atlas_2based()  hrepulsion()   repulsion()
    net.toggle_physics(True)
    net.set_edge_smooth(
        "dynamic"
    )  # dynamic, continuous, discrete, diagonalCross, straightCross, horizontal, vertical, curvedCW, curvedCCW, cubicBezier
    net.set_options(graph_options)
    # net.show_buttons()
    html_graph = net.generate_html()
    return html_graph


############################################################
## Page formating
############################################################

# radio button formatting in line
st.write(
    "<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>", unsafe_allow_html=True
)

st.write("\n")
st.subheader("Factual Question-Answering")
text_option = st.radio("Select Example", ["Insert Query", EXAMPLE_1, EXAMPLE_2, EXAMPLE_3, EXAMPLE_4])
st.write("\n")
if text_option == EXAMPLE_1:
    text_area_input = st.text_input(PLACEHOLDER_QUERY, EXAMPLE_1)
elif text_option == EXAMPLE_2:
    text_area_input = st.text_input(PLACEHOLDER_QUERY, EXAMPLE_2)
elif text_option == EXAMPLE_3:
    text_area_input = st.text_input(PLACEHOLDER_QUERY, EXAMPLE_3)
elif text_option == EXAMPLE_4:
    text_area_input = st.text_input(PLACEHOLDER_QUERY, EXAMPLE_4)
else:
    text_area_input = st.text_input(PLACEHOLDER_QUERY, "")

if st.button("Search") and text_area_input != "":
    with st.spinner("Executing Search ..."):
        pipeline_response = request_pipeline(text_area_input)
    if pipeline_response["status"] == "error":
        st.error("Error while executing Search.")
    else:
        st.text_area("Answer of LLM", value=pipeline_response["answer_llm"], height=180)
        st.text_area("Answer of Graph", value=pipeline_response["answer_graph"], height=180)

        st.write("\n")
        st.markdown("#### **Evidence**")
        st.caption("\n\nCypher Query:")
        st.code(pipeline_response["query"], language="cypher")
        st.caption("Cypher Response:")
        st.code(pipeline_response["response"], language="cypher")
        st.caption("\n\nRelevant Subgraph:")
        html_graph_req = generate_graph(pipeline_response["graph"], send_request=True)
        components.html(html_graph_req, height=550)
        st.text_area("Graph Summary", value=pipeline_response["graph_summary"], height=180)
        st.write("\n")
        st.caption("\n\nJSON Data:")
        with st.expander("Show JSON"):
            pipeline_response["graph"] = pipeline_response["graph"].model_dump(mode="json")
            st.json(pipeline_response)
