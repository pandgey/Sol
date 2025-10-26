import os

# Model Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"

# Paths
PDF_PATH = "/content/qlora_paper.pdf"
VECTORSTORE_PATH = "vectorstore.db"

# Prompt Template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the provided context only to answer the following question:
<context>
{context}
</context>
Question: {input}
"""