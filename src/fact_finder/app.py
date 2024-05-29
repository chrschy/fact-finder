import sys

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from dotenv import load_dotenv
from pyvis.network import Network

import fact_finder.config.primekg_config as graph_config
import fact_finder.config.simple_config as llm_config
from fact_finder.ui.graph_conversion import Subgraph
from fact_finder.ui.util import PipelineOptions, request_pipeline
from fact_finder.utils import load_chat_model

load_dotenv()


############################################################
## Constants
############################################################
PLACEHOLDER_QUERY = "Please insert your query"

EXAMPLE_1 = "What are the phenotypes associated with cardioacrofacial dysplasia?"
EXAMPLE_2 = "What are the genes responsible for psoriasis?"
EXAMPLE_3 = "Which diseases involve PINK1?"
EXAMPLE_4 = "How many drugs against epilepsy are available?"
EXAMPLE_5 = "Which medications have more off-label uses than approved indications?"

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
        st.session_state.llm_chain = llm_config.build_chain(chat_model, sys.argv)
        st.session_state.neo4j_rag_chain = graph_config.build_chain(chat_model, True, sys.argv)
        st.session_state.neo4j_chain = graph_config.build_chain(chat_model, False, sys.argv)
        st.session_state.rag_chain = llm_config.build_rag_chain(chat_model, sys.argv)
        st.session_state.summary_chain = graph_config.build_chain_summary(chat_model, sys.argv)


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


def generate_graph(graph: Subgraph, send_request=False, enable_dynamics: bool = True):
    net = Network(
        height="530px", width="100%"
    )  # , bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)
    for node in graph.nodes:
        color = COLORS["query"] if node.in_query else COLORS["answer"] if node.in_answer else COLORS["default"]
        net.add_node(node.id, label=node.name, title=node.type, mass=1, shape="ellipse", color=color)

    for edge in graph.edges:
        color = COLORS["answer"] if edge.in_answer else COLORS["edge"]
        net.add_edge(edge.source, edge.target, width=1, title=edge.name, arrowStrikethrough=False, color=color)

    if enable_dynamics:
        net.force_atlas_2based()  # barnes_hut()  force_atlas_2based()  hrepulsion()   repulsion()
        net.toggle_physics(True)
        net.set_edge_smooth(
            "dynamic"
        )  # dynamic, continuous, discrete, diagonalCross, straightCross, horizontal, vertical, curvedCW, curvedCCW, cubicBezier
        net.set_options(graph_options)
        # net.show_buttons()
    else:
        net.repulsion()

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
text_option = st.radio("Select Example", ["Insert Query", EXAMPLE_1, EXAMPLE_2, EXAMPLE_3, EXAMPLE_4, EXAMPLE_5])
st.write("\n")

if text_option != "Insert Query":
    text_area_input = st.text_input(PLACEHOLDER_QUERY, text_option)
else:
    text_area_input = st.text_input(PLACEHOLDER_QUERY, "")


pipelines_selected = st.multiselect(
    label="Select Pipelines",
    options=[
        PipelineOptions.LLM.value,
        PipelineOptions.GRAPH.value,
        PipelineOptions.DOC.value,
        PipelineOptions.GRAPH_DOC.value,
        PipelineOptions.GRAPH_SUM.value,
    ],
    default=[
        PipelineOptions.LLM.value,
        PipelineOptions.GRAPH.value
    ],
)


if st.button("Search") and text_area_input != "" and len(pipelines_selected) > 0:
    with st.spinner("Executing Search ..."):
        pipeline_response = request_pipeline(text_area_input, pipelines_selected, st.session_state)
    if pipeline_response.status == "error":
        st.error("Error while executing Search.")
    else:
        if PipelineOptions.LLM.value in pipelines_selected:
            st.markdown("#### **LLM only**")
            st.text_area("Answer using LLM", value=pipeline_response.llm_answer, height=180)

        if PipelineOptions.GRAPH_DOC.value in pipelines_selected:
            st.markdown("#### **LLM with Facts**")
            st.text_area("Answer using Graph & Documents", value=pipeline_response.graph_rag_answer, height=180)

        if PipelineOptions.DOC.value in pipelines_selected:
            st.markdown("#### **Document Retrieval**")
            st.text_area("Answer using Documents", value=pipeline_response.rag_answer, height=180)
            with st.expander("Show Sources"):
                st.text_input("Keywords", value=pipeline_response.rag_keywords)
                st.text_area("Paragraphs", value=pipeline_response.rag_paragraphs, height=180)
                if pipeline_response.rag_prompt_answer != "":
                    st.text_area("Prompt", value=pipeline_response.rag_prompt_answer, height=180)

        if PipelineOptions.GRAPH.value in pipelines_selected:
            st.markdown("#### **Graph Retrieval**")
            st.text_area("Answer using Graph", value=pipeline_response.graph_answer, height=180)
            st.caption("\n\nRelevant Subgraph:")
            html_graph_req = generate_graph(pipeline_response.graph_subgraph, send_request=True)
            components.html(html_graph_req, height=550)
            with st.expander("Show Evidence"):
                st.caption("\n\nCypher Query:")
                st.code(pipeline_response.graph_query, language="cypher")
                st.caption("Cypher Response:")
                st.code(pipeline_response.graph_response, language="cypher")
                if PipelineOptions.GRAPH_SUM.value in pipelines_selected:
                    st.text_area("Graph Summary", value=pipeline_response.graph_summary, height=180)
                st.write("\n")

                if pipeline_response.graph_expanded_subgraph is not None:
                    st.caption("\n\nExpanded Relevant Subgraph:")
                    html_graph_req = generate_graph(
                        pipeline_response.graph_expanded_subgraph, send_request=True, enable_dynamics=False
                    )
                    components.html(html_graph_req, height=550)
                    if PipelineOptions.GRAPH_SUM.value in pipelines_selected:
                        st.text_area(
                            "Expanded Graph Summary", value=pipeline_response.graph_expanded_summary, height=180
                        )
                    st.write("\n")

                if pipeline_response.graph_prompt_cypher != "":
                    st.text_area("Cypher Prompt", value=pipeline_response.graph_prompt_cypher, height=180)
                if pipeline_response.graph_prompt_answer != "":
                    st.text_area("Answer Prompt", value=pipeline_response.graph_prompt_answer, height=180)

        st.caption("\n\nJSON Data:")
        with st.expander("Show JSON"):
            response_display = pipeline_response.model_dump(mode="json")
            st.json(response_display)
