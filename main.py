from processor.documentProcessor import DocumentProcessor
from processor.contentReader import ContentReader
from processor.externalDriveReader import ExternalDriveReader, interactive_drive_selection
from tmp.vectorStore import VectorStoreManager
from logic.qaChain import QAChain
import config
import os

def setup_from_local_folder(content_dir, vectorstore_path):
    """Setup RAG from local folder."""
    print("=" * 60)
    print("METHOD 1: Loading from LOCAL FOLDER")
    print("=" * 60)
    
    reader = ContentReader(content_dir=content_dir)
    documents = reader.get_all_documents(split_docs=True)
    
    if not documents:
        raise ValueError(f"No documents found in '{content_dir}'")
    
    vector_manager = VectorStoreManager(config.EMBEDDING_MODEL)
    vector_manager.create_vectorstore(documents)
    vector_manager.save_vectorstore(vectorstore_path)
    print(f"✓ Vectorstore saved to {vectorstore_path}")
    
    return vector_manager


def setup_from_external_drive_interactive(vectorstore_path):
    """Setup RAG from external drive using interactive selection."""
    print("=" * 60)
    print("METHOD 2: Loading from EXTERNAL DRIVE (Interactive)")
    print("=" * 60)
    
    # Use interactive selection
    documents = interactive_drive_selection()
    
    if not documents:
        raise ValueError("No documents loaded from external drive")
    
    # Create vectorstore
    vector_manager = VectorStoreManager(config.EMBEDDING_MODEL)
    vector_manager.create_vectorstore(documents)
    vector_manager.save_vectorstore(vectorstore_path)
    print(f"✓ Vectorstore saved to {vectorstore_path}")
    
    return vector_manager


def setup_from_external_drive_direct(drive_path, subfolder=None, 
                                     file_types=None, vectorstore_path=None):
    """Setup RAG from external drive using direct path."""
    print("=" * 60)
    print("METHOD 3: Loading from EXTERNAL DRIVE (Direct Path)")
    print("=" * 60)
    print(f"Drive: {drive_path}")
    if subfolder:
        print(f"Subfolder: {subfolder}")
    if file_types:
        print(f"File types: {file_types}")
    
    reader = ExternalDriveReader()
    documents = reader.read_from_drive(
        drive_path=drive_path,
        subfolder=subfolder,
        file_types=file_types,
        split_docs=True
    )
    
    if not documents:
        raise ValueError("No documents loaded from external drive")
    
    # Create vectorstore
    vector_manager = VectorStoreManager(config.EMBEDDING_MODEL)
    vector_manager.create_vectorstore(documents)
    
    if vectorstore_path:
        vector_manager.save_vectorstore(vectorstore_path)
        print(f"✓ Vectorstore saved to {vectorstore_path}")
    
    return vector_manager


def load_existing_vectorstore(vectorstore_path):
    """Load existing vectorstore (fastest option after first setup)."""
    print("=" * 60)
    print("METHOD 4: Loading EXISTING VECTORSTORE")
    print("=" * 60)
    
    vector_manager = VectorStoreManager(config.EMBEDDING_MODEL)
    vector_manager.load_vectorstore(vectorstore_path)
    print("✓ Vectorstore loaded successfully")
    
    return vector_manager


def main():
    """Main execution function with multiple storage options."""
    
    print("\n" + "=" * 60)
    print("RAG SYSTEM - Document Loading Options")
    print("=" * 60)
    print("\nChoose your document source:")
    print("1. Local folder (content/)")
    print("2. External drive - Interactive (USB, HDD, etc.)")
    print("3. External drive - Direct path")
    print("4. Load existing vectorstore")
    print("=" * 60)
    
    choice = input("\nSelect option (1-4): ").strip()
    
    try:
        # Choose setup method based on user input
        if choice == '1':
            # Local folder
            vector_manager = setup_from_local_folder("content", config.VECTORSTORE_PATH)
        
        elif choice == '2':
            # External drive - Interactive
            vector_manager = setup_from_external_drive_interactive(config.VECTORSTORE_PATH)
        
        elif choice == '3':
            # External drive - Direct path
            print("\nEnter the drive path:")
            print("  Windows example: E:\\")
            print("  macOS example: /Volumes/MyUSB")
            print("  Linux example: /media/user/USB_DRIVE")
            
            drive_path = input("\nDrive path: ").strip()
            
            subfolder = input("Subfolder (press Enter to use root): ").strip()
            subfolder = subfolder if subfolder else None
            
            file_filter = input("File types filter (e.g., .pdf,.docx or Enter for all): ").strip()
            file_types = [t.strip() for t in file_filter.split(',')] if file_filter else None
            
            vector_manager = setup_from_external_drive_direct(
                drive_path=drive_path,
                subfolder=subfolder,
                file_types=file_types,
                vectorstore_path=config.VECTORSTORE_PATH
            )
        
        elif choice == '4':
            # Load existing
            vector_manager = load_existing_vectorstore(config.VECTORSTORE_PATH)
        
        else:
            print("Invalid option!")
            return
        
        # Create retriever
        retriever = vector_manager.get_retriever()
        
        # Set up QA chain
        print("\nSetting up QA chain...")
        qa_system = QAChain(config.LLM_MODEL, config.PROMPT_TEMPLATE)
        qa_system.create_chain(retriever)
        
        # Interactive Q&A loop
        print("\n" + "=" * 60)
        print("RAG System Ready! Ask questions about your documents.")
        print("Commands:")
        print("  - Type your question to get an answer")
        print("  - 'quit' or 'exit' to stop")
        print("=" * 60 + "\n")
        
        while True:
            question = input("\n📝 Your Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print(f"\n🔍 Processing: {question}")
            print("-" * 60)
            
            answer = qa_system.get_answer(question)
            print(f"\n💡 Answer:\n{answer}")
            print("-" * 60)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# ============================================================================
# QUICK START EXAMPLES (uncomment to use directly)
# ============================================================================

# Example 1: Local folder (simplest)
# vector_manager = setup_from_local_folder("content", config.VECTORSTORE_PATH)

# Example 2: External USB drive (Windows)
# reader = ExternalDriveReader()
# documents = reader.read_from_drive("E:\\", subfolder="Documents", file_types=['.pdf'])

# Example 3: External USB drive (macOS)
# reader = ExternalDriveReader()
# documents = reader.read_from_drive("/Volumes/MyUSB", file_types=['.pdf', '.docx'])

# Example 4: External USB drive (Linux)
# reader = ExternalDriveReader()
# documents = reader.read_from_drive("/media/username/USB_DRIVE")

# Example 5: Load existing vectorstore (fastest)
# vector_manager = load_existing_vectorstore(config.VECTORSTORE_PATH)