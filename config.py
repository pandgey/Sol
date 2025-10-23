import os

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "Your Key")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Model Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-3.5-turbo"

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