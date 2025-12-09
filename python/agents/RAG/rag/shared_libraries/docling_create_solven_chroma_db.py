import os
import sys
import logging
import nest_asyncio
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
import argparse
from pathlib import Path
import time

# Apply nest_asyncio to allow async loops in scripts (Crucial for some LlamaIndex components)
nest_asyncio.apply()

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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
        logger.info(f"Configuring Google GenAI with embedding model: {model_name}")
        genai.configure(api_key=api_key)
        logger.debug(f"Google GenAI configured successfully with API key (first 8 chars): {api_key[:8]}...")

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        logger.debug(f"Generating embedding for text of length {len(text)} chars")
        result = genai.embed_content(
            model=self.model_name, content=text, task_type="retrieval_document"
        )
        embedding = result["embedding"]
        logger.debug(f"Generated embedding vector of dimension {len(embedding)}")
        return embedding

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
        logger.info("Initializing Docling DocumentConverter with default settings")
        logger.debug("Using default PDF backend (pdfium) with automatic format detection")
        self.converter = DocumentConverter()
        logger.info("Docling DocumentConverter initialized successfully")

    def parse_pdf_to_markdown(
        self, file_path: Path
    ) -> Optional[str]:
        """
        Parses a single PDF file using Docling and returns markdown content.
        Returns None if parsing fails.
        """
        start_time = time.time()
        try:
            logger.info(f"Starting PDF parsing: {file_path.name}")
            logger.debug(f"Full path: {file_path}")
            
            # Convert the PDF to a ConversionResult
            logger.debug("Calling DocumentConverter.convert()...")
            result = self.converter.convert(file_path)

            if result.document is None:
                logger.error(f"Failed to parse {file_path.name}: No document generated")
                return None

            logger.debug(f"Document conversion completed for {file_path.name}")
            
            # Export the parsed document to Markdown
            logger.debug("Exporting document to Markdown format...")
            markdown_content = result.document.export_to_markdown()
            
            elapsed = time.time() - start_time
            markdown_size = len(markdown_content)
            logger.info(
                f"Successfully parsed {file_path.name} in {elapsed:.2f}s "
                f"(Markdown size: {markdown_size:,} chars)"
            )
            logger.debug(f"First 200 chars of markdown: {markdown_content[:200]}...")
            return markdown_content

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Failed to parse {file_path.name} after {elapsed:.2f}s: {e}",
                exc_info=True
            )
            return None

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parses a single PDF file using Docling and returns a LlamaIndex Document."""
        logger.debug(f"Loading data from: {file.name}")
        markdown_content = self.parse_pdf_to_markdown(file)

        if markdown_content is None:
            logger.warning(f"Skipping {file.name} - parsing returned None")
            return []

        metadata = {"file_path": str(file)}
        if extra_info:
            metadata.update(extra_info)
            logger.debug(f"Added extra metadata: {extra_info}")

        logger.debug(f"Created LlamaIndex Document with {len(metadata)} metadata fields")
        # Return as a single LlamaIndex Document with the full markdown
        return [Document(text=markdown_content, metadata=metadata)]


def get_existing_files(collection) -> set[str]:
    """
    Retrieves the set of source file paths that have already been ingested.
    Assumes that the 'file_path' metadata field is populated in the ChromaDB collection.
    """
    try:
        logger.debug("Fetching existing files from ChromaDB collection...")
        # Fetch all metadata from the collection.
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", []) or []
        logger.debug(f"Retrieved {len(metadatas)} metadata entries from collection")

        existing_files = set()
        for idx, meta in enumerate(metadatas):
            if meta and "file_path" in meta:
                # Normalize path to ensure consistent comparison
                normalized_path = str(Path(meta["file_path"]).resolve())
                existing_files.add(normalized_path)
                logger.debug(f"Entry {idx}: {Path(meta['file_path']).name}")

        logger.info(f"Found {len(existing_files)} unique files already in collection")
        return existing_files
    except Exception as e:
        logger.warning(f"Could not retrieve existing files from Chroma: {e}", exc_info=True)
        return set()


def main():
    script_start_time = time.time()
    logger.info("="*80)
    logger.info("Starting Docling PDF Ingestion Script")
    logger.info("="*80)
    
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into ChromaDB with Google Embeddings and Docling PDF parsing."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and re-ingest EVERYTHING.",
    )
    args = parser.parse_args()
    
    logger.info(f"Script arguments: reset={args.reset}")

    load_dotenv()
    logger.info("Environment variables loaded from .env file")

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  PDF Input Directory: {PDF_DIRECTORY}")
    logger.info(f"  ChromaDB Path: {CHROMA_DB_PATH}")
    logger.info(f"  Reset Mode: {args.reset}")

    # --- Google Cloud Embeddings ---
    # Using Google's 'text-embedding-004' (768 dim) for semantic search.
    logger.info("Initializing Google Cloud Embeddings (text-embedding-004)...")

    embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
    Settings.embed_model = embed_model
    Settings.llm = None
    logger.info("LlamaIndex Settings configured:")
    logger.info(f"  Embedding Model: {embed_model.model_name}")
    logger.info("  LLM: None (embedding-only mode)")

    # --- Docling PDF Parser ---
    # Using Docling for advanced PDF understanding with layout, reading order, tables, etc.
    logger.info("Initializing Docling Document Converter...")
    docling_processor = DoclingPDFProcessor()

    # --- ChromaDB Configuration ---
    logger.info("-" * 80)
    logger.info("ChromaDB Setup")
    logger.info("-" * 80)
    logger.info(f"Initializing ChromaDB at {CHROMA_DB_PATH}...")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    logger.info("ChromaDB client initialized successfully")

    collection_name = "eiopa_insurance_google_004"
    logger.info(f"Target collection name: {collection_name}")

    # Handle Reset
    if args.reset:
        logger.warning(
            f"RESET flag detected. Deleting collection '{collection_name}'..."
        )
        try:
            db.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
        except Exception as e:
            logger.info(f"Collection did not exist or could not be deleted: {e}")

    # Create or retrieve collection for the Google embedding model (768 dims)
    logger.debug(f"Getting or creating collection: {collection_name}")
    chroma_collection = db.get_or_create_collection(collection_name)
    
    # Get collection stats
    try:
        count = chroma_collection.count()
        logger.info(f"Collection '{collection_name}' currently has {count} documents")
    except Exception as e:
        logger.debug(f"Could not get collection count: {e}")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- Loading & Ingestion ---
    logger.info("-" * 80)
    logger.info("File Discovery & Processing")
    logger.info("-" * 80)

    # 1. Identify all PDF files in the input directory
    input_dir = Path(PDF_DIRECTORY)
    logger.info(f"Scanning input directory: {input_dir}")
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    all_pdf_files = list(input_dir.glob("*.pdf"))
    logger.info(f"Found {len(all_pdf_files)} PDF files in input directory")
    
    if not all_pdf_files:
        logger.warning("No PDF files found in input directory.")
        return
    
    # Log all discovered files
    for idx, pdf in enumerate(all_pdf_files, 1):
        logger.debug(f"  {idx}. {pdf.name} ({pdf.stat().st_size:,} bytes)")

    # 2. Identify which files need to be processed
    files_to_process = []

    if args.reset:
        logger.info("Reset mode: Processing ALL files")
        files_to_process = all_pdf_files
    else:
        existing_files = get_existing_files(chroma_collection)
        logger.info(f"Filtering out {len(existing_files)} already-ingested files...")

        for pdf_file in all_pdf_files:
            # Resolve to absolute path for comparison
            abs_path = str(pdf_file.resolve())
            if abs_path not in existing_files:
                files_to_process.append(pdf_file)
                logger.debug(f"  NEW: {pdf_file.name}")
            else:
                logger.debug(f"  SKIP (exists): {pdf_file.name}")

    if not files_to_process:
        logger.info(
            "No NEW files to process. Exiting. (Use --reset to force re-ingestion)"
        )
        logger.info(f"Total script runtime: {time.time() - script_start_time:.2f}s")
        return

    logger.info(f"Files to process: {len(files_to_process)}/{len(all_pdf_files)}")
    logger.info("-" * 80)
    logger.info("Starting PDF Parsing")
    logger.info("-" * 80)

    # Parse all PDFs using Docling
    documents = []
    parse_start_time = time.time()
    successful_parses = 0
    failed_parses = 0
    
    for idx, pdf_file in enumerate(files_to_process, 1):
        logger.info(f"Processing file {idx}/{len(files_to_process)}: {pdf_file.name}")
        doc_list = docling_processor.load_data(pdf_file)
        
        if doc_list:
            documents.extend(doc_list)
            successful_parses += 1
            logger.info(f"  ✓ Success ({successful_parses}/{idx} completed)")
        else:
            failed_parses += 1
            logger.warning(f"  ✗ Failed ({failed_parses}/{idx} failures)")
    
    parse_elapsed = time.time() - parse_start_time
    logger.info("-" * 80)
    logger.info("PDF Parsing Summary")
    logger.info("-" * 80)
    logger.info(f"Total files processed: {len(files_to_process)}")
    logger.info(f"Successfully parsed: {successful_parses}")
    logger.info(f"Failed to parse: {failed_parses}")
    logger.info(f"Total parsing time: {parse_elapsed:.2f}s")
    logger.info(f"Average time per file: {parse_elapsed/len(files_to_process):.2f}s")

    if not documents:
        logger.error("No documents were successfully parsed. Exiting.")
        return

    # --- Markdown Node Splitting ---
    logger.info("-" * 80)
    logger.info("Document Chunking")
    logger.info("-" * 80)
    logger.info("Splitting documents using Markdown structure...")
    node_start_time = time.time()
    
    node_parser = MarkdownNodeParser()
    nodes = node_parser.get_nodes_from_documents(documents)
    
    node_elapsed = time.time() - node_start_time
    logger.info(f"Generated {len(nodes)} semantic nodes in {node_elapsed:.2f}s")
    logger.info(f"Average nodes per document: {len(nodes)/len(documents):.1f}")
    
    # Log sample node info
    if nodes:
        sample_node = nodes[0]
        logger.debug(f"Sample node text length: {len(sample_node.text)} chars")
        logger.debug(f"Sample node metadata: {sample_node.metadata}")

    # Indexing
    logger.info("-" * 80)
    logger.info("Embedding & Indexing")
    logger.info("-" * 80)
    logger.info(f"Embedding {len(nodes)} nodes using Google text-embedding-004...")
    logger.info("This may take several minutes depending on the number of nodes.")
    
    index_start_time = time.time()
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    index_elapsed = time.time() - index_start_time
    
    logger.info(f"Indexing completed in {index_elapsed:.2f}s")
    logger.info(f"Average time per node: {index_elapsed/len(nodes):.3f}s")

    # Final summary
    total_elapsed = time.time() - script_start_time
    logger.info("="*80)
    logger.info("Ingestion Complete - Summary")
    logger.info("="*80)
    logger.info(f"Total runtime: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
    logger.info(f"Files processed: {successful_parses}/{len(files_to_process)}")
    logger.info(f"Nodes created: {len(nodes)}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"ChromaDB path: {CHROMA_DB_PATH}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
