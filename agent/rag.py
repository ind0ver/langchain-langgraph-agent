"""RAG knowledge base tools — loads PDFs into ChromaDB and provides search."""
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from config import DOCS_DIRECTORY, PERSIST_DIRECTORY, EMBEDDING_MODEL


# --- Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_K = 3


# --- Build or load the vector store ---
def get_vector_store():
    """Load existing vector store from disk, or create it from PDFs if missing."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # If the database folder exists and has content, just load it
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        print("Loading existing vector store from disk...")
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )

    # Otherwise, build it from scratch
    print(f"Building vector store from PDFs in '{DOCS_DIRECTORY}'...")
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # Load all PDF files from the folder
    loader = DirectoryLoader(
        DOCS_DIRECTORY,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} PDF pages.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    print(f"Vector store saved to '{PERSIST_DIRECTORY}'.")
    return vector_store


# Initialize the vector store once when the module loads
vector_store = get_vector_store()


# --- Tool for the agent ---
@tool
def search_vector_store(query: str) -> str:
    """Search the PIX RPA documentation PDFs for information.
    Use this tool whenever you need to answer questions about PIX platform
    features, configuration, scripts, workflows, or troubleshooting.
    The tool returns the most relevant text passages from the official docs.

    Args:
        query: The search query — be specific about what you want to find.
    """
    docs = vector_store.similarity_search(query, k=SEARCH_K)
    if not docs:
        return "No relevant information found in the Pix documentation."

    results = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        results.append(f"--- Source: {source} ---\n{doc.page_content}")

    return "\n\n".join(results)