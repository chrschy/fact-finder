import os
import asyncio
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import json
import requests
from pyvis.network import Network
from dotenv import load_dotenv
import fact_finder.config.primekg_config as graph_config
import fact_finder.config.simple_config as llm_config
from fact_finder.utils import concatenate_with_headers, load_chat_model


load_dotenv()


############################################################
## Constants
############################################################
PLACEHOLDER_QUERY = "Please insert your query"

# EXAMPLE_1 = "Which biomarkers are associated with a diagnosis of atopic dermatitis?"
EXAMPLE_1 = "What are the phenotypes associated with cardioacrofacial dysplasia?"
EXAMPLE_2 = "What are the genes responsible for psoriasis?"
EXAMPLE_3 = "Which diseases involve PINK1?"
EXAMPLE_4 = "Which drugs could be promising candidates to treat Parkinson?"

max_width_str = f"max-width: 65%;"
style = "<style>mark.entity { display: inline-block }</style>"
graph_options = ''' 
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
'''

COLORS = {
    "edge": "#aaa",
    "default": "#eee",
    "query": "#ffbe00",
    "answer": "#46d000"
}

TYPE_MAPPING = {
    "d": "disease",
    "g": "gene/protein",
    "p": "phenotypes "
}


############################################################
## Page configuration
############################################################
st.set_page_config(
    page_title="Fact Finder", 
    menu_items={
         'Get Help': None,
         'Report a bug': None,
         'About': "## Demonstrator Fact Finder"
        }
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
        image = Image.open('src/img/logo.png')
        st.image(image, width=300)
with col2:
        st.header(" ")
        st.header("Fact Finder", divider='grey')


############################################################
## initialize pipeline
############################################################
if "counter" not in st.session_state:
    st.session_state.counter = 0
    with st.spinner('Initializing Chains...'):
        chat_model = load_chat_model()
        st.session_state.neo4j_chain = graph_config.build_chain(chat_model)
        st.session_state.llm_chain = llm_config.build_chain(chat_model)


############################################################
## Function definitions
############################################################

def get_html(html: str, legend=False):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1.5rem">{}</div>"""
    if legend: WRAPPER = """<div style="overflow-x: auto; padding: 1rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)

async def call_neo4j(message):
    return st.session_state.neo4j_chain(message)

async def call_llm(message):
    return st.session_state.llm_chain(message)

async def call_chains(message):
    results = await asyncio.gather(call_neo4j(message), call_llm(message))
    print(results)
    return results

# async def call_chains(message):
#     results = await asyncio.gather(st.session_state.neo4j_chain.ainvoke(message), st.session_state.llm_chain.ainvoke(message))
#     return results


def convert_subgraph(graph: [], result: str):
    graph_converted = {"entities": [], "relations": []}
    result_ents = [res_val for res in result for res_val in res.values()]

    idx_rel = 0
    for triplet in graph:
        trip = [value for key, value in triplet.items() if type(value) is tuple][0]
        head_type = [key for key, value in triplet.items() if value == trip[0]][0]
        tail_type = [key for key, value in triplet.items() if value == trip[2]][0]
        if trip[0]["index"] not in [ent["id"] for ent in graph_converted["entities"]]:
            graph_converted["entities"].append({"id": trip[0]["index"], "type": head_type, "name": trip[0]["name"], "query": False, "answer": trip[0]["name"] in result_ents})
        if trip[2]["index"] not in [ent["id"] for ent in graph_converted["entities"]]:
            graph_converted["entities"].append({"id": trip[2]["index"], "type": tail_type, "name": trip[2]["name"], "query": False, "answer": trip[2]["name"] in result_ents})
        graph_converted["relations"].append({"id": idx_rel, "type": trip[1], "name": trip[1], "source": trip[0]["index"], "target": trip[2]["index"], "answer": trip[2]["name"] in result_ents})
        idx_rel += 1

    return graph_converted


def request_pipeline(text_data):
    try:
        results = asyncio.run(call_chains(text_data))
        return {
            "status": "success",
            "query": results[0]["intermediate_steps"][0]["query"], 
            "response": results[0]["intermediate_steps"][1]["context"], 
            "answer_graph": results[0]["result"], 
            "answer_llm": results[1]["text"], 
            "graph": convert_subgraph(results[0]["sub_graph"], results[0]["intermediate_steps"][1]["context"]), 
            "graph_neo4j": results[0]["sub_graph"]
        }
    except requests.exceptions.ConnectionError as e:
        print(e)
        return {"status":"error","query":"", "response":"", "answer_graph": "", "answer_llm": "", "graph":{}, "graph_neo4j": []}


def generate_graph(graph, send_request=False):
    net = Network(height="530px", width="100%")#, bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)

    for node in graph["entities"]:
        color = COLORS["query"] if node["query"] else COLORS["answer"] if node["answer"] else COLORS["default"]
        net.add_node(node["id"], label=node["name"], title=node["type"], mass=1, shape="ellipse", color=color) 

    for edge in graph["relations"]:
        color = COLORS["answer"] if edge["answer"] else COLORS["edge"]
        net.add_edge(edge["source"], edge["target"], width=1, title=edge["name"], arrowStrikethrough=False, color=color)

    net.force_atlas_2based()    # barnes_hut()  force_atlas_2based()  hrepulsion()   repulsion()
    net.toggle_physics(True)
    net.set_edge_smooth("dynamic")  # dynamic, continuous, discrete, diagonalCross, straightCross, horizontal, vertical, curvedCW, curvedCCW, cubicBezier
    net.set_options(graph_options)
    # net.show_buttons()
    html_graph = net.generate_html()
    return html_graph


############################################################
## Page formating
############################################################

# radio button formatting in line
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)

st.write('\n')
st.subheader("Factual Question-Answering")
text_option = st.radio("Select Example", ["Insert Query", EXAMPLE_1, EXAMPLE_2, EXAMPLE_3, EXAMPLE_4])
st.write('\n')
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
    with st.spinner('Executing Search ...'):
        pipeline_response = request_pipeline(text_area_input)
    if pipeline_response["status"] == "error":
        st.error("Error while executing Search.")
    else:
        st.text_area("Answer of LLM", value=pipeline_response["answer_llm"], height=180)
        st.text_area("Answer of Graph", value=pipeline_response["answer_graph"], height=180)

        st.write('\n')
        st.markdown("#### **Evidence**")
        st.caption("\n\nCypher Query:")
        st.code(pipeline_response["query"], language="cypher")
        st.caption("Cypher Response:")
        st.code(pipeline_response["response"], language="cypher")
        st.caption("\n\nRelevant Subgraph:")
        html_graph_req = generate_graph(pipeline_response["graph"], send_request=True)
        components.html(html_graph_req, height=550)
        st.write('\n')
        st.caption("\n\nJSON Data:")
        with st.expander("Show JSON"):
            st.json(pipeline_response)
