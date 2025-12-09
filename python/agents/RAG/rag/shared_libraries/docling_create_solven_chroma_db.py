import os
import sys
import logging
import nest_asyncio
from dotenv import load_dotenv
from typing import Any
import argparse
from pathlib import Path

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
    import google.generativeai as genai
    from llama_index.core import (
        VectorStoreIndex,
        StorageContext,
        Settings,
        SimpleDirectoryReader,
    )
    from llama_parse import LlamaParse
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.core.embeddings import BaseEmbedding
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    sys.exit(1)


class GoogleGenAIEmbedding(BaseEmbedding):
    """Custom Embedding class using google-generativeai SDK directly."""

    def __init__(
        self, model_name: str = "models/text-embedding-004", **kwargs: Any
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required.")
        genai.configure(api_key=api_key)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        result = genai.embed_content(
            model=self.model_name, content=text, task_type="retrieval_document"
        )
        return result["embedding"]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


def get_existing_files(collection) -> set[str]:
    """
    Retrieves the set of source file paths that have already been ingested.
    Assumes that the 'file_path' metadata field is populated by LlamaIndex/SimpleDirectoryReader.
    """
    try:
        # Fetch all metadata. simpler than scrolling if the dataset isn't massive.
        # For massive datasets, you'd want to paginate or use a different tracking mechanism.
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", []) or []

        existing_files = set()
        for meta in metadatas:
            if meta and "file_path" in meta:
                # Normalize path to ensure consistent comparison
                existing_files.add(str(Path(meta["file_path"]).resolve()))

        return existing_files
    except Exception as e:
        logger.warning(f"Could not retrieve existing files from Chroma: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into ChromaDB with SOTA embedding and parsing."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and re-ingest EVERYTHING.",
    )
    args = parser.parse_args()

    load_dotenv()

    # --- SOTA UPGRADE 1: The Embedder ---
    # Switching to Google's 'text-embedding-004' (768 dim).
    # This runs in the cloud and is extremely fast.
    logger.info("Initializing Google Cloud Embeddings (text-embedding-004)...")

    embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
    Settings.embed_model = embed_model
    Settings.llm = None

    # --- SOTA UPGRADE 2: The Parser (LlamaParse) ---
    logger.info("Initializing LlamaParse...")

    # Get API Key from .env (LLAMA_CLOUD_API_KEY)
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        logger.error(
            "LLAMA_CLOUD_API_KEY not found in .env. Required for SOTA parsing."
        )
        return

    llama_parse_parser = LlamaParse(
        result_type="markdown",  # Output markdown to preserve headers/tables
        verbose=True,
        language="en",
    )

    # Use SimpleDirectoryReader but mapping .pdf to our advanced parser
    file_extractor = {".pdf": llama_parse_parser}

    # --- SOTA UPGRADE 3: ChromaDB Configuration ---

    logger.info(f"Initializing ChromaDB at {CHROMA_DB_PATH}...")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection_name = "eiopa_insurance_google_004"

    # Handle Reset
    if args.reset:
        logger.warning(
            f"RESET flag detected. Deleting collection '{collection_name}'..."
        )
        try:
            db.delete_collection(collection_name)
            logger.info("Collection deleted.")
        except Exception:
            logger.info("Collection did not exist or could not be deleted.")

    # We use a NEW collection for the Google model (768 dims)
    chroma_collection = db.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- Loading & Ingestion ---

    # 1. Identify all target files
    input_dir = Path(PDF_DIRECTORY)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    all_pdf_files = list(input_dir.glob("*.pdf"))
    if not all_pdf_files:
        logger.info("No PDF files found in input directory.")
        return

    # 2. Identify files to skip (unless resetting)
    files_to_process = []

    if args.reset:
        files_to_process = all_pdf_files
    else:
        existing_files = get_existing_files(chroma_collection)
        logger.info(f"Found {len(existing_files)} files already in valid index.")

        for pdf_file in all_pdf_files:
            # Resolve to absolute path for comparison
            abs_path = str(pdf_file.resolve())
            if abs_path not in existing_files:
                files_to_process.append(pdf_file)
            else:
                logger.debug(f"Skipping existing file: {pdf_file.name}")

    if not files_to_process:
        logger.info(
            "No NEW files to process. Exiting. (Use --reset to force re-ingestion)"
        )
        return

    logger.info(f"Starting ingestion for {len(files_to_process)} files...")
    logger.info(f"Parsing PDFs with AI Vision (LlamaParse) from {PDF_DIRECTORY}...")

    # Load data using the file_extractor map for SPECIFIC files
    # Convert Path objects to strings for SimpleDirectoryReader
    input_files_str = [str(p) for p in files_to_process]

    reader = SimpleDirectoryReader(
        input_files=input_files_str, file_extractor=file_extractor
    )
    documents = reader.load_data()
    logger.info(f"Parsed {len(documents)} document pages/sections.")

    # --- SOTA UPGRADE 4: Markdown Node Splitting ---
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
