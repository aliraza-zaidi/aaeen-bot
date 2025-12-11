class Config:    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "pakistan_constitution"
    PDF_PATH = "constitution.pdf"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 3
    LLM_MODEL = "llama-3.1-8b-instant"
    LLM_TEMPERATURE = 0
    THREAD_ID = "thread-1"