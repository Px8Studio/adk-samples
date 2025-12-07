import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Checks if required packages are installed."""
    try:
        import chromadb
        from llama_index.core import (
            VectorStoreIndex,
            StorageContext,
            Settings,
        )
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return True
    except ImportError as e:
        logger.error(f"Missing dependency for local RAG: {e}")
        logger.error(
            "Please install required packages: uv add chromadb llama-index-vector-stores-chroma llama-index-embeddings-huggingface"
        )
        return False


# Global/module-level cache for the query engine to avoid reloading on every call
_QUERY_ENGINE = None


def _get_or_initialize_query_engine():
    global _QUERY_ENGINE
    if _QUERY_ENGINE:
        return _QUERY_ENGINE

    if not check_dependencies():
        raise ImportError("Missing dependencies for local RAG.")

    import chromadb
    from llama_index.core import VectorStoreIndex, StorageContext, Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    chroma_db_path = os.environ.get("CHROMA_DB_PATH")
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "eiopa_insurance_bge_m3")

    if not chroma_db_path:
        raise ValueError("CHROMA_DB_PATH environment variable is not set.")

    logger.info(f"Initializing Local RAG with ChromaDB at {chroma_db_path}...")

    # 1. Setup Embeddings (Must match ingestion model!)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", trust_remote_code=True)
    Settings.embed_model = embed_model
    # We are only doing retrieval, so we don't strictly need an LLM here for the index itself,
    # but the query engine might use it for response synthesis if we let it.
    # However, we want to return raw text chunks for the agent to use, so we will use the retriever.
    Settings.llm = None

    # 2. Connect to ChromaDB
    db = chromadb.PersistentClient(path=chroma_db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 3. Load Index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    # 4. Create Retriever
    # Using similarity_top_k from env or default to 3
    top_k = int(os.environ.get("RAG_SIMILARITY_TOP_K", 3))

    # We use the retriever directly to get nodes
    retriever = index.as_retriever(similarity_top_k=top_k)

    _QUERY_ENGINE = retriever
    return _QUERY_ENGINE


def retrieve_chroma_documentation(query: str) -> str:
    """Use this tool to retrieve documentation and reference materials for the question from the local ChromaDB corpus.

    Args:
        query: The query string to search within the corpus.
    """
    try:
        retriever = _get_or_initialize_query_engine()
        nodes = retriever.retrieve(query)

        if not nodes:
            return "No relevant documentation found in local corpus."

        response_text = ""
        for i, node in enumerate(nodes, 1):
            # node.node.get_text() gives the content
            # node.score gives the similarity score
            score = f"{node.score:.2f}" if node.score is not None else "N/A"
            metadata = node.node.metadata or {}
            source_file = metadata.get("file_name", "Unknown File")

            response_text += (
                f"--- Source {i} (Score: {score}, File: {source_file}) ---\n"
            )
            response_text += f"{node.node.get_text()}\n\n"

        return response_text

    except Exception as e:
        logger.error(f"Error retrieving from local ChromaDB: {e}")
        return f"Error retrieving from local ChromaDB: {e}"


def list_chroma_sources() -> list[str]:
    """Returns a list of unique documents in the local ChromaDB corpus."""
    try:
        import chromadb

        chroma_db_path = os.environ.get("CHROMA_DB_PATH")
        collection_name = os.environ.get(
            "CHROMA_COLLECTION_NAME", "eiopa_insurance_bge_m3"
        )

        if not chroma_db_path:
            return ["CHROMA_DB_PATH not set."]

        db = chromadb.PersistentClient(path=chroma_db_path)
        collection = db.get_collection(collection_name)

        # Fetch all metadata to find unique files
        # Warning: This scales linearly with corpus size. For large corpora, this should be cached or optimized.
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])

        unique_files = set()
        for m in metadatas:
            if m and "file_name" in m:
                unique_files.add(m["file_name"])

        if not unique_files:
            count = collection.count()
            return [
                f"Collection contains {count} chunks, but no 'file_name' metadata found."
            ]

        return sorted(list(unique_files))

    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        return [f"Error listing sources: {e}"]


def get_chroma_file_metadata(file_name: str) -> str:
    """Returns metadata for a specific file in the local ChromaDB corpus.

    Args:
        file_name: The name of the file to get metadata for.

    Returns:
        A formatted string containing the file's metadata found in the vector store.
    """
    try:
        import chromadb

        chroma_db_path = os.environ.get("CHROMA_DB_PATH")
        collection_name = os.environ.get(
            "CHROMA_COLLECTION_NAME", "eiopa_insurance_bge_m3"
        )

        if not chroma_db_path:
            return "CHROMA_DB_PATH not set."

        db = chromadb.PersistentClient(path=chroma_db_path)
        collection = db.get_collection(collection_name)

        # Query for items where file_name matches
        # Note: ChromaDB filtering syntax
        results = collection.get(
            where={"file_name": file_name}, limit=1, include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])
        if not metadatas or not metadatas[0]:
            return f"No metadata found for file: {file_name}"

        # Return the first matching metadata dict
        # We assume all chunks for the same file share largely the same document-level metadata
        return str(metadatas[0])

    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        return f"Error getting file metadata: {e}"
