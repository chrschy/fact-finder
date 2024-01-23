import os

import chainlit as cl
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PW = os.getenv("NEO4J_PW", "opensesame")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None


@cl.on_chat_start
async def on_chat_start():
    graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PW)
    model = ChatOpenAI(streaming=True, temperature=0, api_key=OPENAI_API_KEY)
    chain = GraphCypherQAChain.from_llm(model, graph=graph, verbose=True)
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    res = await chain.arun(query=message.content, callbacks=[cl.LangchainCallbackHandler()])
    await cl.Message(content=res).send()
