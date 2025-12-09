# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Optimized PDF ingestion pipeline using Docling's advanced features.

This version uses:
- Docling's HybridChunker for structure-aware, token-aligned chunking
- Formula enrichment for LaTeX extraction
- Advanced table structure extraction
- OCR for scanned content
- Code enrichment
- Aligned tokenization with embedding model
"""

from __future__ import annotations

import os
import sys
import logging
import nest_asyncio
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
import argparse
from pathlib import Path
import time

# Apply nest_asyncio to allow async loops in scripts
nest_asyncio.apply()

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
PDF_DIRECTORY = r"C:\Users\rjjaf\_Projects\solven\backend\data\eiopa\insurance\input"
CHROMA_DB_PATH = r"C:\Users\rjjaf\_Projects\solven\backend\data\chroma_db"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768
EMBEDDING_MAX_TOKENS = 2048  # Google text-embedding-004 limit

# Global Imports
try:
    import chromadb
    import google.generativeai as genai
    from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    
    # Docling imports
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TesseractCliOcrOptions
    )
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.datamodel.base_models import InputFormat
    from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
    from docling.chunking import HybridChunker
    
    # Tokenizer for alignment with embedding model
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
    
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Install required packages: pip install docling transformers")
    sys.exit(1)


class GoogleGenAIEmbedding(BaseEmbedding):
    """Custom Embedding class using google-generativeai SDK directly."""

    def __init__(
        self, model_name: str = EMBEDDING_MODEL_NAME, **kwargs: Any
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required.")
        logger.info(f"Configuring Google GenAI with embedding model: {model_name}")
        genai.configure(api_key=api_key)
        logger.debug(f"API key configured (first 8 chars): {api_key[:8]}...")

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


class OptimizedDoclingProcessor:
    """
    Enhanced PDF processor using Docling's advanced features:
    - Formula enrichment (LaTeX extraction)
    - Advanced table structure extraction
    - OCR for scanned content
    - Code enrichment
    - Structure-aware chunking with HybridChunker
    """

    def __init__(self, enable_ocr: bool = True, enable_formulas: bool = True,
                 enable_code: bool = True, table_mode: str = "accurate"):
        """
        Initialize the optimized Docling processor.
        
        Args:
            enable_ocr: Enable OCR for scanned documents
            enable_formulas: Enable formula extraction (LaTeX)
            enable_code: Enable code block enrichment
            table_mode: 'accurate' or 'fast' for table extraction
        """
        logger.info("="*80)
        logger.info("Initializing Optimized Docling Processor")
        logger.info("="*80)
        
        # Configure pipeline options for technical documents
        pipeline_options = PdfPipelineOptions()
        
        # OCR Configuration
        pipeline_options.do_ocr = enable_ocr
        if enable_ocr:
            logger.info("✓ OCR enabled for scanned content")
            
            # RapidOCR is recommended: 20-100x faster than Tesseract
            # Performance: ~10-50ms per page vs 1-2s for Tesseract
            try:
                from docling.datamodel.pipeline_options import RapidOcrOptions
                pipeline_options.ocr_options = RapidOcrOptions(
                    backend="onnxruntime",  # CPU-optimized for server deployment
                    text_score=0.5,  # Default confidence threshold
                    lang=["english", "chinese"]  # RapidOCR default languages
                )
                logger.info("  Engine: RapidOCR (20-100x faster than Tesseract)")
                logger.info("  Backend: onnxruntime (CPU-optimized)")
                logger.info("  Note: Install with: uv add rapidocr-onnxruntime")
            except ImportError:
                logger.warning("  RapidOCR not available, falling back to TesseractCli")
                logger.warning("  For production: uv add rapidocr-onnxruntime")
                pipeline_options.ocr_options = TesseractCliOcrOptions(
                    force_full_page_ocr=False,  # Only OCR when needed
                    lang=["eng"]  # Languages to recognize
                )
                logger.info("  Engine: TesseractCli (fallback)")
                logger.info("  Note: Requires: choco install tesseract (Windows)")
        else:
            logger.info("✗ OCR disabled")
        
        # Table Structure Extraction (CRITICAL for technical docs)
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST
        )
        
        if table_mode == "accurate":
            logger.info("✓ Table extraction enabled (mode: ACCURATE)")
            logger.info("  Speed: ~500-1000ms per table (prioritizes quality)")
            logger.info("  Best for: Insurance policies, complex nested tables, financial docs")
        else:
            logger.info("✓ Table extraction enabled (mode: FAST)")
            logger.info("  Speed: ~50-100ms per table (prioritizes speed)")
            logger.info("  Best for: High-volume processing, simple tables")
        
        # Formula Enrichment (CRITICAL for mathematical content)
        pipeline_options.do_formula_enrichment = enable_formulas
        if enable_formulas:
            logger.info("✓ Formula enrichment enabled (LaTeX extraction)")
        else:
            logger.info("✗ Formula enrichment disabled")
        
        # Code Enrichment
        pipeline_options.do_code_enrichment = enable_code
        if enable_code:
            logger.info("✓ Code enrichment enabled")
        else:
            logger.info("✗ Code enrichment disabled")
        
        # Image Classification (optional but useful)
        pipeline_options.do_picture_classification = True
        logger.info("✓ Picture classification enabled")
        
        # Accelerator options (use GPU if available)
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.AUTO  # Auto-detect GPU/CPU
        )
        logger.info("✓ Accelerator set to AUTO (GPU if available)")
        
        # Initialize DocumentConverter with optimized settings
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=DoclingParseV4DocumentBackend
                )
            }
        )
        logger.info("✓ DocumentConverter initialized with optimized settings")
        logger.info("="*80)

    def parse_pdf(self, file_path: Path) -> Optional[Any]:
        """
        Parse a PDF file and return the DoclingDocument.
        
        Returns:
            DoclingDocument or None if parsing fails
        """
        start_time = time.time()
        try:
            logger.info(f"Starting PDF parsing: {file_path.name}")
            logger.debug(f"Full path: {file_path}")
            
            # Convert the PDF
            logger.debug("Calling DocumentConverter.convert()...")
            result = self.converter.convert(file_path)
            
            if result.document is None:
                logger.error(f"Failed to parse {file_path.name}: No document generated")
                return None
            
            elapsed = time.time() - start_time
            
            # Log document statistics
            doc = result.document
            logger.info(
                f"Successfully parsed {file_path.name} in {elapsed:.2f}s"
            )
            logger.info(f"  Pages: {len(doc.pages)}")
            logger.info(f"  Tables: {len(doc.tables)}")
            logger.info(f"  Pictures: {len(doc.pictures)}")
            
            # Log first bit of content as markdown for validation
            markdown_preview = doc.export_to_markdown()[:200]
            logger.debug(f"Markdown preview: {markdown_preview}...")
            
            return doc
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Failed to parse {file_path.name} after {elapsed:.2f}s: {e}",
                exc_info=True
            )
            return None


def chunk_documents_with_hybrid_chunker(
    docling_docs: List[tuple[Path, Any]], 
    tokenizer_model: str = "thenlper/gte-large-en"
) -> List[Document]:
    """
    Chunk DoclingDocuments using HybridChunker for optimal results.
    
    Args:
        docling_docs: List of (file_path, DoclingDocument) tuples
        tokenizer_model: HuggingFace tokenizer to use (should match embedding model)
    
    Returns:
        List of LlamaIndex Document objects with rich metadata
    """
    logger.info("="*80)
    logger.info("Hybrid Chunking Configuration")
    logger.info("="*80)
    logger.info(f"Tokenizer model: {tokenizer_model}")
    logger.info(f"  Rationale: GTE models are semantically aligned with Google embeddings")
    logger.info(f"  Alternative: 'sentence-transformers/all-mpnet-base-v2' for lighter weight")
    logger.info(f"Max tokens per chunk: {EMBEDDING_MAX_TOKENS}")
    
    # Initialize tokenizer aligned with embedding model
    # CRITICAL: Google doesn't expose text-embedding-004/005 tokenizer to developers
    # GTE (Google Text Embedding) models are explicitly trained for semantic alignment
    # and publish their tokenizers. This is better than using BERT as a proxy.
    logger.debug("Loading HuggingFace tokenizer...")
    logger.debug(f"  Note: IMPORTANT - Tokenizer-embedding alignment directly impacts RAG quality")
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=EMBEDDING_MAX_TOKENS
    )
    logger.info("✓ Tokenizer initialized")
    
    # Initialize HybridChunker
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True  # Merge small adjacent chunks with same context
    )
    logger.info("✓ HybridChunker initialized")
    logger.info("="*80)
    
    # Chunk all documents
    all_chunks = []
    chunk_start_time = time.time()
    
    for file_path, docling_doc in docling_docs:
        logger.info(f"Chunking document: {file_path.name}")
        doc_chunk_start = time.time()
        
        try:
            # Chunk the document
            chunks = list(chunker.chunk(docling_doc))
            
            doc_chunk_elapsed = time.time() - doc_chunk_start
            logger.info(
                f"  Created {len(chunks)} chunks in {doc_chunk_elapsed:.2f}s "
                f"({doc_chunk_elapsed/len(chunks):.3f}s per chunk)"
            )
            
            # Convert to LlamaIndex documents with rich metadata
            for idx, chunk in enumerate(chunks):
                # Extract comprehensive metadata for RAG optimization
                # Phase 1: Core metadata (always included)
                metadata = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
                
                # Phase 2: Hierarchical context (for document structure awareness)
                if chunk.meta.headings:
                    metadata["headings"] = chunk.meta.headings
                    logger.debug(f"  Chunk {idx}: Headings: {chunk.meta.headings}")
                
                if chunk.meta.captions:
                    metadata["captions"] = chunk.meta.captions
                    logger.debug(f"  Chunk {idx}: Captions: {chunk.meta.captions}")
                
                # Phase 3: Content type indicators (enables content-aware filtering)
                if chunk.meta.doc_items:
                    item_labels = [item.label.value for item in chunk.meta.doc_items]
                    metadata["doc_item_types"] = item_labels
                    logger.debug(f"  Chunk {idx}: Item types: {item_labels}")
                    
                    # Content indicators for advanced filtering
                    metadata["has_tables"] = any(label == "table" for label in item_labels)
                    metadata["has_code"] = any(label == "code" for label in item_labels)
                    metadata["has_formulas"] = any(label == "formula" for label in item_labels)
                    metadata["has_lists"] = any(label in ["ordered_list", "unordered_list"] for label in item_labels)
                
                # Phase 4: Text metrics (for quality indicators)
                metadata["text_length_chars"] = len(chunk.text)
                
                # Try to count tokens for the contextualized version
                try:
                    contextualized_text = chunker.contextualize(chunk=chunk)
                    metadata["text_with_context_length_chars"] = len(contextualized_text)
                    logger.debug(
                        f"  Chunk {idx}: Raw={len(chunk.text)} chars, "
                        f"Contextualized={len(contextualized_text)} chars"
                    )
                except Exception as e:
                    logger.debug(f"  Chunk {idx}: Could not contextualize text: {e}")
                
                logger.debug(f"  Chunk {idx}: Metadata keys: {list(metadata.keys())}")
                
                all_chunks.append(Document(text=chunk.text, metadata=metadata))
            
        except Exception as e:
            logger.error(f"Failed to chunk {file_path.name}: {e}", exc_info=True)
    
    chunk_elapsed = time.time() - chunk_start_time
    logger.info("="*80)
    logger.info("Chunking Summary")
    logger.info("="*80)
    logger.info(f"Total documents chunked: {len(docling_docs)}")
    logger.info(f"Total chunks created: {len(all_chunks)}")
    logger.info(f"Average chunks per document: {len(all_chunks)/len(docling_docs):.1f}")
    logger.info(f"Total chunking time: {chunk_elapsed:.2f}s")
    logger.info("="*80)
    
    return all_chunks


def get_existing_files(collection) -> set[str]:
    """
    Retrieves the set of source file paths that have already been ingested.
    """
    try:
        logger.debug("Fetching existing files from ChromaDB collection...")
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", []) or []
        logger.debug(f"Retrieved {len(metadatas)} metadata entries from collection")

        existing_files = set()
        for idx, meta in enumerate(metadatas):
            if meta and "file_path" in meta:
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
    logger.info("Starting OPTIMIZED Docling PDF Ingestion Script")
    logger.info("="*80)
    
    parser = argparse.ArgumentParser(
        description="Optimized PDF ingestion with Docling's advanced features and HybridChunker."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and re-ingest EVERYTHING.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR (faster, but won't process scanned content).",
    )
    parser.add_argument(
        "--no-formulas",
        action="store_true",
        help="Disable formula enrichment (faster, but won't extract LaTeX).",
    )
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Disable code enrichment (faster).",
    )
    parser.add_argument(
        "--table-mode",
        choices=["accurate", "fast"],
        default="accurate",
        help="Table extraction mode: 'accurate' (slower, better) or 'fast'.",
    )
    args = parser.parse_args()
    
    logger.info(f"Script arguments:")
    logger.info(f"  Reset: {args.reset}")
    logger.info(f"  OCR: {not args.no_ocr}")
    logger.info(f"  Formulas: {not args.no_formulas}")
    logger.info(f"  Code: {not args.no_code}")
    logger.info(f"  Table mode: {args.table_mode}")

    load_dotenv()
    logger.info("Environment variables loaded from .env file")

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  PDF Input Directory: {PDF_DIRECTORY}")
    logger.info(f"  ChromaDB Path: {CHROMA_DB_PATH}")
    logger.info(f"  Embedding Model: {EMBEDDING_MODEL_NAME}")
    logger.info(f"  Embedding Dimension: {EMBEDDING_DIMENSION}")
    logger.info(f"  Max Tokens per Chunk: {EMBEDDING_MAX_TOKENS}")

    # Initialize Google Cloud Embeddings
    logger.info("-" * 80)
    logger.info("Embedding Model Setup")
    logger.info("-" * 80)
    embed_model = GoogleGenAIEmbedding(model_name=EMBEDDING_MODEL_NAME)
    Settings.embed_model = embed_model
    Settings.llm = None
    logger.info("✓ LlamaIndex Settings configured")

    # Initialize Docling Processor
    logger.info("-" * 80)
    docling_processor = OptimizedDoclingProcessor(
        enable_ocr=not args.no_ocr,
        enable_formulas=not args.no_formulas,
        enable_code=not args.no_code,
        table_mode=args.table_mode
    )

    # ChromaDB Configuration
    logger.info("-" * 80)
    logger.info("ChromaDB Setup")
    logger.info("-" * 80)
    logger.info(f"Initializing ChromaDB at {CHROMA_DB_PATH}...")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    logger.info("✓ ChromaDB client initialized")

    collection_name = "eiopa_insurance_google_004"
    logger.info(f"Target collection name: {collection_name}")

    # Handle Reset
    if args.reset:
        logger.warning(f"RESET flag detected. Deleting collection '{collection_name}'...")
        try:
            db.delete_collection(collection_name)
            logger.info(f"✓ Collection '{collection_name}' deleted")
        except Exception as e:
            logger.info(f"Collection did not exist: {e}")

    # Create or retrieve collection
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

    # File Discovery & Processing
    logger.info("-" * 80)
    logger.info("File Discovery & Processing")
    logger.info("-" * 80)

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
    
    # Log discovered files
    for idx, pdf in enumerate(all_pdf_files, 1):
        logger.debug(f"  {idx}. {pdf.name} ({pdf.stat().st_size:,} bytes)")

    # Identify files to process
    files_to_process = []
    if args.reset:
        logger.info("Reset mode: Processing ALL files")
        files_to_process = all_pdf_files
    else:
        existing_files = get_existing_files(chroma_collection)
        logger.info(f"Filtering out {len(existing_files)} already-ingested files...")
        for pdf_file in all_pdf_files:
            abs_path = str(pdf_file.resolve())
            if abs_path not in existing_files:
                files_to_process.append(pdf_file)
                logger.debug(f"  NEW: {pdf_file.name}")
            else:
                logger.debug(f"  SKIP (exists): {pdf_file.name}")

    if not files_to_process:
        logger.info("No NEW files to process. Exiting. (Use --reset to force re-ingestion)")
        logger.info(f"Total script runtime: {time.time() - script_start_time:.2f}s")
        return

    logger.info(f"Files to process: {len(files_to_process)}/{len(all_pdf_files)}")
    
    # Parse PDFs
    logger.info("-" * 80)
    logger.info("PDF Parsing with Docling")
    logger.info("-" * 80)
    
    docling_documents = []
    parse_start_time = time.time()
    successful_parses = 0
    failed_parses = 0
    
    for idx, pdf_file in enumerate(files_to_process, 1):
        logger.info(f"Processing file {idx}/{len(files_to_process)}: {pdf_file.name}")
        docling_doc = docling_processor.parse_pdf(pdf_file)
        
        if docling_doc:
            docling_documents.append((pdf_file, docling_doc))
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

    if not docling_documents:
        logger.error("No documents were successfully parsed. Exiting.")
        return

    # Chunk documents with HybridChunker
    logger.info("-" * 80)
    logger.info("Chunking Documents with HybridChunker")
    logger.info("-" * 80)
    
    documents = chunk_documents_with_hybrid_chunker(docling_documents, tokenizer_model="thenlper/gte-large")
    
    if not documents:
        logger.error("No chunks created. Exiting.")
        return

    # Indexing
    logger.info("-" * 80)
    logger.info("Embedding & Indexing")
    logger.info("-" * 80)
    logger.info(f"Embedding {len(documents)} chunks using {EMBEDDING_MODEL_NAME}...")
    logger.info("This may take several minutes...")
    
    index_start_time = time.time()
    
    # Note: With HybridChunker, documents are already optimal chunks
    # We can index them directly without additional node parsing
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    
    index_elapsed = time.time() - index_start_time
    logger.info(f"✓ Indexing completed in {index_elapsed:.2f}s")
    logger.info(f"Average time per chunk: {index_elapsed/len(documents):.3f}s")

    # Final summary
    total_elapsed = time.time() - script_start_time
    logger.info("="*80)
    logger.info("Ingestion Complete - Summary")
    logger.info("="*80)
    logger.info(f"Total runtime: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
    logger.info(f"Files processed: {successful_parses}/{len(files_to_process)}")
    logger.info(f"Chunks created: {len(documents)}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"ChromaDB path: {CHROMA_DB_PATH}")
    logger.info("="*80)
    logger.info("✅ SUCCESS! Optimized ingestion complete.")
    logger.info("="*80)


if __name__ == "__main__":
    main()
