from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# For openai key
import os
os.environ["OPENAI_API_KEY"] = "Your Key"

# load a PDF  
loader = PyPDFLoader("/content/qlora_paper.pdf")
documents = loader.load()

# Split text 
text = RecursiveCharacterTextSplitter().split_documents(documents)

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", 
encode_kwargs={"normalize_embeddings": True})

# Create a vectorstore
vectorstore = FAISS.from_documents(text, embeddings)

# Save the documents and embeddings
vectorstore.save_local("vectorstore.db")

# Create retriever
retriever = vectorstore.as_retriever()

# Load the llm 
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Define prompt template
template = """
You are an assistant for question-answering tasks.
Use the provided context only to answer the following question:

<context>
{context}
</context>

Question: {input}
"""

# Create a prompt template
prompt = ChatPromptTemplate.from_template(template)

# Create a chain 
doc_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, doc_chain)

# User query 
response = chain.invoke({"input": "what is Qlora?"})

# Get the Answer only
response['answer']