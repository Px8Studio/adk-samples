import os
import sys
import logging
import nest_asyncio
from dotenv import load_dotenv

# Apply nest_asyncio to allow async loops in scripts (Crucial for LlamaParse)
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check input directory
PDF_DIRECTORY = r"C:\Users\rjjaf\_Projects\solven\backend\data\eiopa\insurance\input"
CHROMA_DB_PATH = r"C:\Users\rjjaf\_Projects\solven\backend\data\chroma_db"

# Global Imports
try:
    import chromadb
    from llama_index.core import (
        VectorStoreIndex,
        StorageContext,
        Settings,
        SimpleDirectoryReader,
    )
    from llama_parse import LlamaParse
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.node_parser import MarkdownNodeParser
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    sys.exit(1)


def check_dependencies():
    """Checks if required packages are installed."""
    # Already checked via global imports which fail fast
    pass


def main():
    load_dotenv()

    # --- SOTA UPGRADE 1: The Embedder ---
    # Replaced 'all-MiniLM-L6-v2' (384 dim) with 'BAAI/bge-m3' (1024 dim).
    # BGE-M3 is currently one of the best open-source models for complex retrieval.
    # It supports a larger context window (8192 tokens) suitable for long insurance clauses.
    logger.info("Initializing SOTA local embeddings (BGE-M3)...")

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", trust_remote_code=True)
    Settings.embed_model = embed_model
    Settings.llm = None

    # --- SOTA UPGRADE 2: The Parser (LlamaParse) ---
    # Standard parsers break tables. LlamaParse converts PDFs to Markdown,
    # preserving the structural integrity of financial tables in EIOPA docs.
    logger.info("Initializing LlamaParse...")

    # Get API Key from .env (LLAMA_CLOUD_API_KEY)
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        logger.error(
            "LLAMA_CLOUD_API_KEY not found in .env. Required for SOTA parsing."
        )
        return

    parser = LlamaParse(
        result_type="markdown",  # Output markdown to preserve headers/tables
        verbose=True,
        language="en",
    )

    # Use SimpleDirectoryReader but mapping .pdf to our advanced parser
    file_extractor = {".pdf": parser}

    # --- SOTA UPGRADE 3: ChromaDB Configuration ---

    logger.info(f"Initializing ChromaDB at {CHROMA_DB_PATH}...")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # We use a collection name specific to the model to avoid dimension conflicts
    # (MiniLM is 384 dims, BGE-M3 is 1024 dims - they cannot mix!)
    chroma_collection = db.get_or_create_collection("eiopa_insurance_bge_m3")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- Loading & Ingestion ---
    logger.info(f"Parsing PDFs with AI Vision (LlamaParse) from {PDF_DIRECTORY}...")

    # Load data using the file_extractor map
    reader = SimpleDirectoryReader(PDF_DIRECTORY, file_extractor=file_extractor)
    documents = reader.load_data()
    logger.info(f"Parsed {len(documents)} document pages/sections.")

    # --- SOTA UPGRADE 4: Markdown Node Splitting ---
    # Because we parsed to Markdown, we use a specific splitter that respects
    # headers and table boundaries rather than splitting in mid-sentence.

    logger.info("Splitting documents based on Markdown structure...")
    node_parser = MarkdownNodeParser()
    nodes = node_parser.get_nodes_from_documents(documents)
    logger.info(f"Generated {len(nodes)} semantic nodes.")

    # Indexing
    logger.info("Embedding and indexing nodes...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    logger.info("Success! SOTA Ingestion Complete.")


if __name__ == "__main__":
    main()
