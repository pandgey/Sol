from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain


class QAChain:
    """Handles the question-answering chain setup and execution."""
    
    def __init__(self, llm_model, prompt_template):
        self.llm = ChatOpenAI(model_name=llm_model)
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = None
    
    def create_chain(self, retriever):
        """Create the retrieval QA chain."""
        doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.chain = create_retrieval_chain(retriever, doc_chain)
        return self.chain
    
    def query(self, question):
        """Ask a question and get an answer."""
        if self.chain is None:
            raise ValueError("Chain not created. Call create_chain first.")
        
        response = self.chain.invoke({"input": question})
        return response
    
    def get_answer(self, question):
        """Get only the answer text from a query."""
        response = self.query(question)
        return response.get('answer', '')