import logging
import os
import sys
from typing import Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Checks if required packages are installed."""
    try:
        import chromadb
        import google.generativeai as genai
        from llama_index.core import (
            VectorStoreIndex,
            StorageContext,
            Settings,
        )
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core.embeddings import BaseEmbedding

        return True
    except ImportError as e:
        logger.error(f"Missing dependency for local RAG: {e}")
        return False


# Global/module-level cache
_QUERY_ENGINE = None


class GoogleGenAIEmbedding:
    """Custom Embedding class using google-generativeai SDK directly (Simplified for Tool)."""

    # Note: We duplicate this class here to avoid module import issues.
    # In a proper package structure, this would be in a shared utility.

    def __init__(
        self, model_name: str = "models/text-embedding-004", **kwargs: Any
    ) -> None:
        self.model_name = model_name
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required.")
        genai.configure(api_key=api_key)

    # LlamaIndex expects these methods if we duck-type or wrap it properly
    # But for simplicity in the tool, we just need to pass it to Settings.embed_model
    # IF we inherit from BaseEmbedding. Since BaseEmbedding requires pydantic,
    # and we want to avoid complex inheritance in a simple tool file if possible,
    # let's do it properly by importing BaseEmbedding.
    pass


def _get_custom_google_embedding_class():
    from llama_index.core.embeddings import BaseEmbedding
    import google.generativeai as genai

    class InternalGoogleGenAIEmbedding(BaseEmbedding):
        def __init__(
            self, model_name: str = "models/text-embedding-004", **kwargs: Any
        ) -> None:
            super().__init__(model_name=model_name, **kwargs)
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is required.")
            genai.configure(api_key=api_key)

        def _get_query_embedding(self, query: str) -> List[float]:
            result = genai.embed_content(
                model=self.model_name, content=query, task_type="retrieval_query"
            )
            return result["embedding"]

        def _get_text_embedding(self, text: str) -> List[float]:
            result = genai.embed_content(
                model=self.model_name, content=text, task_type="retrieval_document"
            )
            return result["embedding"]

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str) -> List[float]:
            return self._get_text_embedding(text)

    return InternalGoogleGenAIEmbedding


def _get_or_initialize_query_engine():
    global _QUERY_ENGINE
    if _QUERY_ENGINE:
        return _QUERY_ENGINE

    if not check_dependencies():
        raise ImportError("Missing dependencies for local RAG.")

    import chromadb
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore

    chroma_db_path = os.environ.get("CHROMA_DB_PATH")
    collection_name = os.environ.get(
        "CHROMA_COLLECTION_NAME", "eiopa_insurance_google_004"
    )

    if not chroma_db_path:
        raise ValueError("CHROMA_DB_PATH environment variable is not set.")

    logger.info(f"Initializing Local RAG with ChromaDB at {chroma_db_path}...")

    # 1. Setup Google Embeddings
    GoogleEmbeddingClass = _get_custom_google_embedding_class()
    embed_model = GoogleEmbeddingClass(model_name="models/text-embedding-004")
    Settings.embed_model = embed_model
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
    top_k = int(os.environ.get("RAG_SIMILARITY_TOP_K", 3))
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
            "CHROMA_COLLECTION_NAME", "eiopa_insurance_google_004"
        )

        if not chroma_db_path:
            return ["CHROMA_DB_PATH not set."]

        db = chromadb.PersistentClient(path=chroma_db_path)
        collection = db.get_collection(collection_name)

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
            "CHROMA_COLLECTION_NAME", "eiopa_insurance_google_004"
        )

        if not chroma_db_path:
            return "CHROMA_DB_PATH not set."

        db = chromadb.PersistentClient(path=chroma_db_path)
        collection = db.get_collection(collection_name)

        results = collection.get(
            where={"file_name": file_name}, limit=1, include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])
        if not metadatas or not metadatas[0]:
            return f"No metadata found for file: {file_name}"

        return str(metadatas[0])

    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        return f"Error getting file metadata: {e}"
