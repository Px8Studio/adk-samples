import os
import sys
import logging
import nest_asyncio
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
import argparse
from pathlib import Path

# Apply nest_asyncio to allow async loops in scripts (Crucial for some LlamaIndex components)
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
        Document,
    )
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.node_parser import MarkdownNodeParser
    # Import docling with DocumentConverter
    from docling.document_converter import DocumentConverter
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


class DoclingPDFProcessor:
    """
    A processor that uses Docling's DocumentConverter to parse PDFs to Markdown.
    Docling provides advanced PDF understanding with layout, reading order, tables, etc.
    """

    def __init__(self):
        """Initialize the Docling DocumentConverter."""
        self.converter = DocumentConverter()

    def parse_pdf_to_markdown(
        self, file_path: Path
    ) -> Optional[str]:
        """
        Parses a single PDF file using Docling and returns markdown content.
        Returns None if parsing fails.
        """
        try:
            logger.info(f"Parsing PDF with Docling: {file_path}...")
            # Convert the PDF to a ConversionResult
            result = self.converter.convert(file_path)

            if result.document is None:
                logger.error(f"Failed to parse {file_path}: No document generated")
                return None

            # Export the parsed document to Markdown
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Successfully parsed {file_path.name}")
            return markdown_content

        except Exception as e:
            logger.error(f"Failed to parse {file_path} with Docling: {e}")
            return None

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parses a single PDF file using Docling and returns a LlamaIndex Document."""
        markdown_content = self.parse_pdf_to_markdown(file)

        if markdown_content is None:
            return []

        metadata = {"file_path": str(file)}
        if extra_info:
            metadata.update(extra_info)

        # Return as a single LlamaIndex Document with the full markdown
        return [Document(text=markdown_content, metadata=metadata)]


def get_existing_files(collection) -> set[str]:
    """
    Retrieves the set of source file paths that have already been ingested.
    Assumes that the 'file_path' metadata field is populated in the ChromaDB collection.
    """
    try:
        # Fetch all metadata from the collection.
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
        description="Ingest PDFs into ChromaDB with Google Embeddings and Docling PDF parsing."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and re-ingest EVERYTHING.",
    )
    args = parser.parse_args()

    load_dotenv()

    # --- Google Cloud Embeddings ---
    # Using Google's 'text-embedding-004' (768 dim) for semantic search.
    logger.info("Initializing Google Cloud Embeddings (text-embedding-004)...")

    embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
    Settings.embed_model = embed_model
    Settings.llm = None

    # --- Docling PDF Parser ---
    # Using Docling for advanced PDF understanding with layout, reading order, tables, etc.
    logger.info("Initializing Docling Document Converter...")
    docling_processor = DoclingPDFProcessor()

    # --- ChromaDB Configuration ---

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

    # Create or retrieve collection for the Google embedding model (768 dims)
    chroma_collection = db.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- Loading & Ingestion ---

    # 1. Identify all PDF files in the input directory
    input_dir = Path(PDF_DIRECTORY)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    all_pdf_files = list(input_dir.glob("*.pdf"))
    if not all_pdf_files:
        logger.info("No PDF files found in input directory.")
        return

    # 2. Identify which files need to be processed
    files_to_process = []

    if args.reset:
        files_to_process = all_pdf_files
    else:
        existing_files = get_existing_files(chroma_collection)
        logger.info(f"Found {len(existing_files)} files already in the collection.")

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
    logger.info(f"Parsing PDFs with Docling from {PDF_DIRECTORY}...")

    # Parse all PDFs using Docling
    documents = []
    for pdf_file in files_to_process:
        doc_list = docling_processor.load_data(pdf_file)
        documents.extend(doc_list)

    if not documents:
        logger.error("No documents were successfully parsed.")
        return

    logger.info(f"Successfully parsed {len(documents)} PDF files.")

    # --- Markdown Node Splitting ---
    logger.info("Splitting documents using Markdown structure...")
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

    logger.info("Success! Docling-based ingestion complete.")


if __name__ == "__main__":
    main()
