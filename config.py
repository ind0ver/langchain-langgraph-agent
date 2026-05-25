# --- LLM Settings ---
BASE_URL = "http://localhost:11434"
MODEL = "gemma4:e4b"

# --- SQL Settings ---
DB_PATH = "chinook_db/Chinook_Sqlite.sqlite"

# --- RAG Settings ---
DOCS_DIRECTORY = "./rag_docs"
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Web Search Settings ---
MAX_WEB_SEARCH_RESULTS = 5
