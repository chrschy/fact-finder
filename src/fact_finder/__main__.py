import chainlit as cl
import fact_finder.config.primekg_config as graph_config
import fact_finder.config.simple_config as llm_config
from dotenv import load_dotenv
from fact_finder.utils import concatenate_with_headers, load_chat_model

load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    chat_model = load_chat_model()
    cl.user_session.set("neo4j_chain", graph_config.build_chain(chat_model))
    cl.user_session.set("llm_chain", llm_config.build_chain(chat_model))


@cl.on_message
async def main(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    llm_chain_res = await llm_chain.ainvoke(input=message.content, callbacks=[cl.LangchainCallbackHandler()])
    neo4j_chain = cl.user_session.get("neo4j_chain")
    neo4j_chain_res = await neo4j_chain.ainvoke(input=message.content, callbacks=[cl.LangchainCallbackHandler()])

    answer = concatenate_with_headers(
        [{"LLM:": llm_chain_res["text"]}, {"Graph:": neo4j_chain_res["result"]}]
    )
    answer = neo4j_chain_res["result"]
    await cl.Message(content=answer).send()
