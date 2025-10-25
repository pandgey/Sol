from processor.documentProcessor import DocumentProcessor
from tmp.vectorStore import VectorStoreManager
from logic.qaChain import QAChain
import config

def setup_rag_system(pdf_path, vectorstore_path):
    """Set up the complete RAG system."""
    
    # Process documents
    print("Processing documents...")
    doc_processor = DocumentProcessor()
    documents = doc_processor.process_pdf(pdf_path)
    print(f"Loaded and split {len(documents)} document chunks")
    
    # Create and save vectorstore
    print("Creating vectorstore...")
    vector_manager = VectorStoreManager(config.EMBEDDING_MODEL)
    vector_manager.create_vectorstore(documents)
    vector_manager.save_vectorstore(vectorstore_path)
    print(f"Vectorstore saved to {vectorstore_path}")
    
    return vector_manager


def load_existing_rag_system(vectorstore_path):
    """Load an existing RAG system from saved vectorstore."""
    
    print("Loading existing vectorstore...")
    vector_manager = VectorStoreManager(config.EMBEDDING_MODEL)
    vector_manager.load_vectorstore(vectorstore_path)
    print("Vectorstore loaded successfully")
    
    return vector_manager


def main():
    """Main execution function."""
    
    # Option 1: Set up new RAG system
    vector_manager = setup_rag_system(config.PDF_PATH, config.VECTORSTORE_PATH)
    
    # Option 2: Load existing vectorstore (uncomment to use)
    # vector_manager = load_existing_rag_system(config.VECTORSTORE_PATH)
    
    # Create retriever
    retriever = vector_manager.get_retriever()
    
    # Set up QA chain
    print("Setting up QA chain...")
    qa_system = QAChain(config.LLM_MODEL, config.PROMPT_TEMPLATE)
    qa_system.create_chain(retriever)
    
    # Ask a question
    question = "what is Qlora?"
    print(f"\nQuestion: {question}")
    answer = qa_system.get_answer(question)
    print(f"Answer: {answer}")
    
    # You can ask more questions
    # response = qa_system.query("another question here")
    # print(response['answer'])


if __name__ == "__main__":
    main()