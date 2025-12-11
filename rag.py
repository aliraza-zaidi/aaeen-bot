from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config

def initialize_vector_store():
    """Initialize and return the vector store with embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=Config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=Config.PERSIST_DIRECTORY
    )
    return vector_store

def vectorize(vector_store):
    """Load PDF into vector store if it's empty."""
    if not vector_store.get()['ids']:
        print("--- Loading PDF ---")
        try:
            loader = PyPDFLoader(Config.PDF_PATH)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(docs)
            vector_store.add_documents(splits)
            print("--- PDF Loaded Successfully ---")
        except FileNotFoundError:
            print(f"--- PDF not found: {Config.PDF_PATH} ---")
        except Exception as e:
            print(f"--- Error loading PDF: {e} ---")
