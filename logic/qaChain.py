from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

class QAChain:
    """Handles the question-answering chain setup and execution."""

    def __init__(self, llm_model, prompt_template):
        print(f"Loading local model: {llm_model}...")
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            device_map="auto",
            torch_dtype="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)
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