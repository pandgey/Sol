from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Handles PDF loading and text splitting."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_pdf(self, pdf_path):
        """Load a PDF file and return documents."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    
    def split_documents(self, documents):
        """Split documents into smaller chunks."""
        return self.text_splitter.split_documents(documents)
    
    def process_pdf(self, pdf_path):
        """Load and split a PDF in one step."""
        documents = self.load_pdf(pdf_path)
        split_docs = self.split_documents(documents)
        return split_docs