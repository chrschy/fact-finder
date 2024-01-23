import os

import chainlit as cl
from langchain.chains import GraphCypherQAChain, LLMChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from prompt_templates import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT, LLM_PROMPT
from dotenv import load_dotenv
from utils import concatenate_with_headers

load_dotenv()

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PW = os.getenv("NEO4J_PW", "opensesame")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None


@cl.on_chat_start
async def on_chat_start():
    graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PW)
    model = ChatOpenAI(model="gpt-4", streaming=False, temperature=0, api_key=OPENAI_API_KEY)  # gpt-3.5-turbo-16k
    neo4j_chain = GraphCypherQAChain.from_llm(model, graph=graph, cypher_prompt=CYPHER_GENERATION_PROMPT,
                                              qa_prompt=CYPHER_QA_PROMPT, verbose=True, return_intermediate_steps=True)
    llm_chain = LLMChain(llm=model, prompt=LLM_PROMPT, verbose=True)
    cl.user_session.set("neo4j_chain", neo4j_chain)
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    llm_chain_res = await llm_chain.ainvoke(input=message.content, callbacks=[cl.LangchainCallbackHandler()])
    neo4j_chain = cl.user_session.get("neo4j_chain")
    neo4j_chain_res = await neo4j_chain.ainvoke(input=message.content, callbacks=[cl.LangchainCallbackHandler()])
    concatenated_answer = concatenate_with_headers(
        [{"LLM:": llm_chain_res['text']}, {"Graph:": neo4j_chain_res["result"]}])
    await cl.Message(content=concatenated_answer).send()
