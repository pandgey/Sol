from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    """Manages vector store creation, saving, and loading."""
    
    def __init__(self, embedding_model, encode_kwargs=None):
        if encode_kwargs is None:
            encode_kwargs = {"normalize_embeddings": True}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs=encode_kwargs
        )
        self.vectorstore = None
    
    def create_vectorstore(self, documents):
        """Create a FAISS vectorstore from documents."""
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self.vectorstore
    
    def save_vectorstore(self, path):
        """Save the vectorstore to disk."""
        if self.vectorstore is None:
            raise ValueError("No vectorstore to save. Create one first.")
        self.vectorstore.save_local(path)
    
    def load_vectorstore(self, path):
        """Load a vectorstore from disk."""
        self.vectorstore = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vectorstore
    
    def get_retriever(self, search_kwargs=None):
        """Get a retriever from the vectorstore."""
        if self.vectorstore is None:
            raise ValueError("No vectorstore available. Create or load one first.")
        
        if search_kwargs is None:
            return self.vectorstore.as_retriever()
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)