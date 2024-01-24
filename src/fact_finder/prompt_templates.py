from langchain_core.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE: str = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_QA_TEMPLATE: str = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Information:
{context}

Question: {question}
Helpful Answer:"""

CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

LLM_PROMPT_TEMPLATE: str = """You are a friendly and engaging conversationalist.
You enjoy talking to customers about various topics, from the weather to hobbies and current events.
You are always ready for a casual conversation and enjoy interacting with customers.

Human: {question}
AI:
"""

LLM_PROMPT = PromptTemplate(
    input_variables=["question"], template=LLM_PROMPT_TEMPLATE
)
