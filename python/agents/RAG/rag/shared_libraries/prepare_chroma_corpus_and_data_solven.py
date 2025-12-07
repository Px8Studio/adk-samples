import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check input directory
PDF_DIRECTORY = r"C:\Users\rjjaf\_Projects\solven\backend\data\eiopa\insurance\input"
CHROMA_DB_PATH = r"C:\Users\rjjaf\_Projects\solven\backend\data\chroma_db"


def check_dependencies():
    """Checks if required pages are installed."""
    try:
        import chromadb
        from llama_index.core import (
            VectorStoreIndex,
            SimpleDirectoryReader,
            StorageContext,
            Settings,
        )
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error(
            "Please install required packages: uv add chromadb llama-index-vector-stores-chroma llama-index-embeddings-huggingface"
        )
        sys.exit(1)


def main():
    check_dependencies()

    # Late import to verify dependencies first
    import chromadb
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        StorageContext,
        Settings,
    )
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    load_dotenv()

    if not os.path.exists(PDF_DIRECTORY):
        logger.error(f"Input directory does not exist: {PDF_DIRECTORY}")
        return

    # 1. Setup Local Embeddings (Free, runs on CPU)
    logger.info("Initializing local HuggingFace embeddings...")
    # 'all-MiniLM-L6-v2' is fast and good for general purpose
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.embed_model = embed_model
    Settings.llm = None  # We don't need an LLM for ingestion

    # 2. Setup ChromaDB (Local Vector Store)
    logger.info(f"Initializing ChromaDB at {CHROMA_DB_PATH}...")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection("eiopa_insurance_corpus")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Load Documents
    logger.info(f"Loading PDFs from {PDF_DIRECTORY}...")
    documents = SimpleDirectoryReader(PDF_DIRECTORY).load_data()
    logger.info(f"Loaded {len(documents)} document chunks.")

    # 4. Ingest into Vector Store
    logger.info("Creating index (this may take a while)...")
    index = VectorStoreIndex.from_documents(
        documents, step=4, storage_context=storage_context, embed_model=embed_model
    )

    logger.info("Success! Documents embedded and stored in local ChromaDB.")
    logger.info(f"Database location: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
