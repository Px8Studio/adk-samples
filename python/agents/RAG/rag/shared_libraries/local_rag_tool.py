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
    logger.info(f"Using Chroma Collection: {collection_name}")

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

    This tool leverages the optimized ingestion pipeline's rich metadata for better context.

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

            # Build source header with enhanced metadata
            response_text += (
                f"--- Source {i} (Score: {score}, File: {source_file}"
            )
            
            # Add page information if available
            page_ranges = metadata.get("page_ranges")
            if page_ranges:
                response_text += f", Pages: {page_ranges}"
            
            # Add chunk position information if available
            chunk_index = metadata.get("chunk_index")
            total_chunks = metadata.get("total_chunks")
            if chunk_index is not None and total_chunks is not None:
                response_text += f", Chunk {chunk_index + 1}/{total_chunks}"
            
            response_text += ") ---\n"
            
            # Add contextual metadata indicators
            content_indicators = []
            if metadata.get("has_tables"):
                content_indicators.append("📊 Contains Tables")
            if metadata.get("has_formulas"):
                content_indicators.append("🔢 Contains Formulas")
            if metadata.get("has_code"):
                content_indicators.append("💻 Contains Code")
            if metadata.get("has_lists"):
                content_indicators.append("📋 Contains Lists")
            
            if content_indicators:
                response_text += f"Indicators: {', '.join(content_indicators)}\n"
            
            # Add hierarchical context (section/heading info)
            headings = metadata.get("headings")
            if headings:
                response_text += f"Section: {headings}\n"
            
            # Add caption information for better understanding
            captions = metadata.get("captions")
            if captions:
                response_text += f"Related Content: {captions}\n"
            
            response_text += "\n"
            response_text += f"{node.node.get_text()}\n\n"

        return response_text

    except Exception as e:
        logger.error(f"Error retrieving from local ChromaDB: {e}")
        return f"Error retrieving from local ChromaDB: {e}"


def list_chroma_sources() -> list[str]:
    """Returns a list of unique documents in the local ChromaDB corpus with summary statistics.
    
    Leverages metadata from the optimized ingestion pipeline to provide file counts and statistics.
    """
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

        # Aggregate by file with statistics
        file_stats = {}
        for m in metadatas:
            if m and "file_name" in m:
                file_name = m["file_name"]
                if file_name not in file_stats:
                    file_stats[file_name] = {
                        "chunks": 0,
                        "has_tables": 0,
                        "has_formulas": 0,
                        "has_code": 0,
                    }
                file_stats[file_name]["chunks"] += 1
                if m.get("has_tables"):
                    file_stats[file_name]["has_tables"] += 1
                if m.get("has_formulas"):
                    file_stats[file_name]["has_formulas"] += 1
                if m.get("has_code"):
                    file_stats[file_name]["has_code"] += 1

        if not file_stats:
            count = collection.count()
            return [
                f"Collection contains {count} chunks, but no 'file_name' metadata found."
            ]

        # Format file list with statistics
        result = []
        result.append(f"\n📚 **Available Sources** ({len(file_stats)} files):\n")
        
        for file_name in sorted(file_stats.keys()):
            stats = file_stats[file_name]
            result.append(
                f"• **{file_name}** ({stats['chunks']} chunks)"
            )
            
            # Add content type summary
            content_types = []
            if stats["has_tables"] > 0:
                content_types.append(f"📊 {stats['has_tables']} w/ tables")
            if stats["has_formulas"] > 0:
                content_types.append(f"🔢 {stats['has_formulas']} w/ formulas")
            if stats["has_code"] > 0:
                content_types.append(f"💻 {stats['has_code']} w/ code")
            
            if content_types:
                result.append(f"  {', '.join(content_types)}")
        
        return result

    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        return [f"Error listing sources: {e}"]


def get_chroma_file_metadata(file_name: str) -> str:
    """Returns comprehensive metadata for a specific file in the local ChromaDB corpus.

    This includes document statistics, content types, PDF properties, and coverage information from the
    optimized ingestion pipeline.

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
            where={"file_name": file_name}, include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])
        if not metadatas:
            return f"No metadata found for file: {file_name}"

        # Aggregate statistics across all chunks of this file
        total_chunks = len(metadatas)
        unique_sections = set()
        unique_pages = set()
        has_tables_count = 0
        has_formulas_count = 0
        has_code_count = 0
        has_lists_count = 0
        total_chars = 0
        
        # Extract PDF properties (same across all chunks of a file)
        pdf_properties = {}
        
        for meta in metadatas:
            if meta:
                # Extract PDF properties from first chunk (same for all)
                if not pdf_properties and "total_pages" in meta:
                    pdf_properties = {
                        "total_pages": meta.get("total_pages"),
                        "file_size_bytes": meta.get("file_size_bytes"),
                        "pdf_title": meta.get("pdf_title"),
                        "pdf_author": meta.get("pdf_author"),
                        "pdf_subject": meta.get("pdf_subject"),
                        "pdf_creator": meta.get("pdf_creator"),
                        "pdf_producer": meta.get("pdf_producer"),
                        "pdf_creation_date": meta.get("pdf_creation_date"),
                        "pdf_mod_date": meta.get("pdf_mod_date"),
                    }
                
                # Aggregate content type stats
                if meta.get("has_tables"):
                    has_tables_count += 1
                if meta.get("has_formulas"):
                    has_formulas_count += 1
                if meta.get("has_code"):
                    has_code_count += 1
                if meta.get("has_lists"):
                    has_lists_count += 1
                
                # Collect sections and page ranges
                if "headings" in meta and meta["headings"]:
                    unique_sections.add(meta["headings"])
                if "page_ranges" in meta and meta["page_ranges"]:
                    unique_pages.add(meta["page_ranges"])
                
                total_chars += meta.get("text_length_chars", 0)

        # Format comprehensive metadata report
        report = f"\n📄 **File: {file_name}**\n\n"
        
        # PDF Properties Section
        if pdf_properties.get("total_pages") or pdf_properties.get("file_size_bytes"):
            report += f"**Document Properties:**\n"
            if pdf_properties.get("total_pages"):
                report += f"  • Total Pages: {pdf_properties['total_pages']}\n"
            if pdf_properties.get("file_size_bytes"):
                size_mb = pdf_properties['file_size_bytes'] / (1024 * 1024)
                report += f"  • File Size: {size_mb:.2f} MB\n"
            if pdf_properties.get("pdf_title"):
                report += f"  • Title: {pdf_properties['pdf_title']}\n"
            if pdf_properties.get("pdf_author"):
                report += f"  • Author: {pdf_properties['pdf_author']}\n"
            if pdf_properties.get("pdf_subject"):
                report += f"  • Subject: {pdf_properties['pdf_subject']}\n"
            if pdf_properties.get("pdf_creator"):
                report += f"  • Creator: {pdf_properties['pdf_creator']}\n"
            if pdf_properties.get("pdf_producer"):
                report += f"  • Producer: {pdf_properties['pdf_producer']}\n"
            if pdf_properties.get("pdf_creation_date"):
                report += f"  • Created: {pdf_properties['pdf_creation_date']}\n"
            if pdf_properties.get("pdf_mod_date"):
                report += f"  • Modified: {pdf_properties['pdf_mod_date']}\n"
            report += "\n"
        
        report += f"**Coverage Statistics:**\n"
        report += f"  • Total Chunks: {total_chunks}\n"
        report += f"  • Total Characters: {total_chars:,}\n"
        report += f"  • Unique Sections: {len(unique_sections)}\n"
        report += f"  • Page Ranges Covered: {len(unique_pages)}\n\n"

        report += f"**Content Types Found:**\n"
        report += f"  • Chunks with Tables: {has_tables_count}/{total_chunks} ({100*has_tables_count/total_chunks:.1f}%)\n"
        report += f"  • Chunks with Formulas: {has_formulas_count}/{total_chunks} ({100*has_formulas_count/total_chunks:.1f}%)\n"
        report += f"  • Chunks with Code: {has_code_count}/{total_chunks} ({100*has_code_count/total_chunks:.1f}%)\n"
        report += f"  • Chunks with Lists: {has_lists_count}/{total_chunks} ({100*has_lists_count/total_chunks:.1f}%)\n\n"

        if unique_sections:
            report += f"**Sections Covered:**\n"
            for section in sorted(unique_sections)[:15]:  # Show first 15
                report += f"  • {section}\n"
            if len(unique_sections) > 15:
                report += f"  • ... and {len(unique_sections) - 15} more sections\n"
            report += "\n"
        
        if unique_pages:
            report += f"**Pages with Content:**\n"
            for page_range in sorted(unique_pages):
                report += f"  • Pages {page_range}\n"

        return report

    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        return f"Error getting file metadata: {e}"


def get_corpus_content_summary() -> str:
    """Returns a comprehensive summary of content types available in the entire corpus.
    
    Useful for understanding what specialized content (tables, formulas, code) is available
    before performing detailed searches.
    
    Returns:
        A formatted string containing corpus-wide content type statistics.
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

        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])

        if not metadatas:
            return "Corpus is empty."

        # Aggregate corpus-wide statistics
        total_chunks = len(metadatas)
        total_chars = 0
        chunks_with_tables = 0
        chunks_with_formulas = 0
        chunks_with_code = 0
        chunks_with_lists = 0
        unique_files = set()
        
        for meta in metadatas:
            if meta:
                if "file_name" in meta:
                    unique_files.add(meta["file_name"])
                if meta.get("has_tables"):
                    chunks_with_tables += 1
                if meta.get("has_formulas"):
                    chunks_with_formulas += 1
                if meta.get("has_code"):
                    chunks_with_code += 1
                if meta.get("has_lists"):
                    chunks_with_lists += 1
                total_chars += meta.get("text_length_chars", 0)

        # Format comprehensive corpus summary
        summary = f"\n📚 **Corpus Content Summary**\n\n"
        summary += f"**Overall Statistics:**\n"
        summary += f"  • Total Files: {len(unique_files)}\n"
        summary += f"  • Total Chunks: {total_chunks}\n"
        summary += f"  • Total Characters: {total_chars:,}\n"
        summary += f"  • Average Chunk Size: {total_chars/total_chunks if total_chunks > 0 else 0:.0f} chars\n\n"

        summary += f"**Content Type Distribution:**\n"
        summary += f"  • 📊 Chunks with Tables: {chunks_with_tables}/{total_chunks} ({100*chunks_with_tables/total_chunks:.1f}%)\n"
        summary += f"  • 🔢 Chunks with Formulas: {chunks_with_formulas}/{total_chunks} ({100*chunks_with_formulas/total_chunks:.1f}%)\n"
        summary += f"  • 💻 Chunks with Code: {chunks_with_code}/{total_chunks} ({100*chunks_with_code/total_chunks:.1f}%)\n"
        summary += f"  • 📋 Chunks with Lists: {chunks_with_lists}/{total_chunks} ({100*chunks_with_lists/total_chunks:.1f}%)\n\n"

        summary += f"**Tip:** Use `retrieve_chroma_documentation()` to search for specific content.\n"
        summary += f"Use `list_chroma_sources()` to see all available files.\n"

        return summary

    except Exception as e:
        logger.error(f"Error getting corpus summary: {e}")
        return f"Error getting corpus summary: {e}"
