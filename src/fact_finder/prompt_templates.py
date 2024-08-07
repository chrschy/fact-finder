from langchain_core.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE: str = """Task: Generate Cypher statement to query a graph database described in the following schema.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
If there is no sensible Cypher statement for the given question and schema, state so and prepend SCHEMA_ERROR to your answer.
Any variables that are returned by the query must have readable names.
Remove modifying adjectives from the entities queried to the graph.

Schema:
{schema}

{predicate_descriptions}

Note: 
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question", "predicate_descriptions"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_QA_TEMPLATE: str = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is a list, include all entries in your response.
If the provided information is empty, say that you don't know the answer.
Provided Information:
{context}

Question: {question}
Helpful Answer:"""
CYPHER_QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE)

LLM_PROMPT_TEMPLATE: str = """You are a friendly and engaging conversationalist.
You enjoy talking to customers about various topics, from the weather to hobbies and current events.
You are always ready for a casual conversation and enjoy interacting with customers.

Human: {question}
AI:
"""
LLM_PROMPT = PromptTemplate(input_variables=["question"], template=LLM_PROMPT_TEMPLATE)

SUBGRAPH_EXTRACTOR_PROMPT_TEMPLATE: str = """
Task: Modify a given Cypher query. The new Cypher query returns all relationships and properties used in the query instead of the answer to the question. It also returns all nodes.
Instructions:
If there is no sensible Cypher statement for the given query, state so and prepend CYPHER_ERROR to your answer.
Only return the modified cypher query, nothing else.
It is very important to also ALWAYS add a variable to the predicate, like so:
WRONG:
MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d, dis, properties(indication)
CORRECT:
MATCH (d:drug)-[i:indication]->(dis:disease) WHERE dis.name = 'epilepsy' RETURN d, dis, i
 
The Cypher query is:
{cypher_query}
"""
SUBGRAPH_EXTRACTOR_PROMPT = PromptTemplate(
    input_variables=["cypher_query"], template=SUBGRAPH_EXTRACTOR_PROMPT_TEMPLATE
)


KEYWORD_PROMPT_TEMPLATE: str = """You are a helpful assistant. You get a user question in natural language. Please transform it into keywords that can be used in semantic scholar keyword search:
Question: {question}
"""
KEYWORD_PROMPT = PromptTemplate(input_variables=["question"], template=KEYWORD_PROMPT_TEMPLATE)


RAG_PROMPT_TEMPLATE: str = """You are a helpful assistant. You get a user question in natural language. Given the following context, please answer the given question based only on the context. Do not hallucinate. If you cannot answer based on the context, say 'The documents do not provide enough information to answer the question.'.
Context: {context}, 
Question: {question}
"""
RAG_PROMPT = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)


COMBINED_QA_TEMPLATE: str = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Note that you are provided information from two different sources.
First, a list of relevant paper abstracts. Second, information extracted from a knowledge graph.
The knowledge graph information is always generated as an answer to the question and is always correct.
Include the information from both parts in your answer.
If the provided information is a list, include all entries in your response.
If the provided information is empty, say that you don't know the answer.
1. Paper abstracts information:
{abstracts}

2. Knowledge graph information:
a) Used cypher query:
{cypher_query}
b) Query result from graph:
{graph_answer}

Question: {question}
Helpful Answer:"""
COMBINED_QA_PROMPT = PromptTemplate(
    input_variables=["abstracts", "graph_answer", "question"], template=COMBINED_QA_TEMPLATE
)


SUBGRAPH_SUMMARY_PROMPT_TEMPLATE: str = """
Verbalize the given triplets of a subgraph to natural text. Use all triplets for the verbalization.
 
Triplets of the subgraph:
{sub_graph}
"""
SUBGRAPH_SUMMARY_PROMPT = PromptTemplate(input_variables=["sub_graph"], template=SUBGRAPH_SUMMARY_PROMPT_TEMPLATE)


LLM_JUDGE_PAIRWISE_PROMPT_TEMPLATE: str = """You are given the role of a judge for factual question answering. Act like a biomedical researcher who attaches great importance to the quality of the answers.
Given the input question and a factual reference, which response of two AI assistants do you prefer: A or B?
Evaluate based on the following criteria:
{criteria}
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
Reason step by step and finally, respond with either [[A]] or [[B]] on its own line.

DATA
----
input: {input}
reference: {reference}
answer assistant A: {prediction}
answer assistant B: {prediction_b}
---
Reasoning:

"""
LLM_JUDGE_PAIRWISE_PROMPT = PromptTemplate(input_variables=["criteria", "input", "reference", "prediction", "prediction_b"], template=LLM_JUDGE_PAIRWISE_PROMPT_TEMPLATE)


LLM_JUDGE_SCORE_PROMPT_TEMPLATE: str = """You are given the role of a judge for factual question answering. Act like a biomedical researcher who attaches great importance to the quality of the answers.
Given the input question, a factual reference and a hypothesis answer, determine if the hypothesis answer meets the following criteria:
{criteria}
Reason step by step and finally, respond with either [[1]] if the hypothesis answer meets the criteria or [[8]] if not on its own line.

DATA
----
input: {input}
reference: {reference}
hypothesis answer: {prediction}
---
Reasoning:

"""
LLM_JUDGE_SCORE_PROMPT = PromptTemplate(input_variables=["criteria", "input", "reference", "prediction"], template=LLM_JUDGE_SCORE_PROMPT_TEMPLATE)

LLM_JUDGE_CRITERIA_CORRECTNESS = {
    "correctness": "The submission is considered to be correct, if the answer only contains facts from the reference and there are no hallucinations."
}
LLM_JUDGE_CRITERIA_COMPLETENESS = {
    "completeness": "The submission is considered to be complete, if the answer contains all facts provided by the reference and no facts are missing."
}

PERSONA_RESEARCHER = {
    "Act like a biomedical researcher who attaches great importance to the quality of the answers."
}
PERSONA_LEGAL_EXPERT = {
    "Act like a security and legal expert. You are extremely concerned about the safety of the products that are developed and sold based on these answers."
}
