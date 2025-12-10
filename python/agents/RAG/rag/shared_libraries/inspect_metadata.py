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
Diagnostic script to inspect what metadata fields are being stored in ChromaDB.

This helps verify that the optimized ingestion pipeline is properly extracting and storing:
- Page numbers/ranges
- Content type indicators (tables, formulas, code, lists)
- PDF properties (author, creation date, etc)
- Section headings
- And more
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def inspect_chroma_metadata():
    """Inspect and print metadata from ChromaDB collection."""
    try:
        import chromadb
        
        chroma_db_path = os.environ.get("CHROMA_DB_PATH")
        collection_name = os.environ.get(
            "CHROMA_COLLECTION_NAME", "eiopa_insurance_google_004"
        )
        
        if not chroma_db_path:
            logger.error("CHROMA_DB_PATH not set")
            return
        
        logger.info(f"Connecting to ChromaDB at: {chroma_db_path}")
        db = chromadb.PersistentClient(path=chroma_db_path)
        collection = db.get_collection(collection_name)
        
        # Get all metadata
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Total chunks: {len(metadatas)}")
        logger.info(f"{'='*80}\n")
        
        if not metadatas:
            logger.warning("No metadata found in collection")
            return
        
        # Inspect first chunk in detail
        first_meta = metadatas[0]
        logger.info("FIRST CHUNK METADATA (detailed inspection):")
        logger.info("-" * 80)
        for key, value in sorted(first_meta.items()):
            logger.info(f"  {key}: {value}")
        logger.info("-" * 80)
        
        # Collect all unique metadata keys across all chunks
        all_keys = set()
        for meta in metadatas:
            all_keys.update(meta.keys())
        
        logger.info(f"\nALL UNIQUE METADATA KEYS ACROSS ALL CHUNKS:")
        logger.info("-" * 80)
        for key in sorted(all_keys):
            logger.info(f"  • {key}")
        logger.info("-" * 80)
        
        # Aggregate statistics
        logger.info(f"\nMETADATA STATISTICS:")
        logger.info("-" * 80)
        
        # Page ranges
        chunks_with_pages = sum(1 for m in metadatas if m.get("page_ranges"))
        logger.info(f"✓ Chunks with page_ranges: {chunks_with_pages}/{len(metadatas)}")
        
        # Headings
        chunks_with_headings = sum(1 for m in metadatas if m.get("headings"))
        logger.info(f"✓ Chunks with headings: {chunks_with_headings}/{len(metadatas)}")
        
        # Content types
        chunks_with_tables = sum(1 for m in metadatas if m.get("has_tables"))
        chunks_with_formulas = sum(1 for m in metadatas if m.get("has_formulas"))
        chunks_with_code = sum(1 for m in metadatas if m.get("has_code"))
        chunks_with_lists = sum(1 for m in metadatas if m.get("has_lists"))
        
        logger.info(f"✓ Chunks with tables: {chunks_with_tables}/{len(metadatas)}")
        logger.info(f"✓ Chunks with formulas: {chunks_with_formulas}/{len(metadatas)}")
        logger.info(f"✓ Chunks with code: {chunks_with_code}/{len(metadatas)}")
        logger.info(f"✓ Chunks with lists: {chunks_with_lists}/{len(metadatas)}")
        
        # PDF properties
        chunks_with_pdf_props = sum(1 for m in metadatas if m.get("total_pages"))
        logger.info(f"✓ Chunks with PDF properties: {chunks_with_pdf_props}/{len(metadatas)}")
        
        if chunks_with_pdf_props > 0:
            sample_props = {}
            for m in metadatas:
                if m.get("total_pages"):
                    sample_props = {
                        "total_pages": m.get("total_pages"),
                        "file_size_bytes": m.get("file_size_bytes"),
                        "pdf_title": m.get("pdf_title"),
                        "pdf_author": m.get("pdf_author"),
                        "pdf_creation_date": m.get("pdf_creation_date"),
                    }
                    break
            logger.info(f"  Sample PDF properties: {sample_props}")
        
        logger.info("-" * 80)
        
        # Show sample of different content
        logger.info(f"\nSAMPLE CHUNKS BY CONTENT TYPE:")
        logger.info("-" * 80)
        
        for idx, meta in enumerate(metadatas[:5]):
            logger.info(f"\nChunk {idx}:")
            logger.info(f"  Pages: {meta.get('page_ranges', 'N/A')}")
            logger.info(f"  Section: {meta.get('headings', 'N/A')}")
            content_types = []
            if meta.get("has_tables"):
                content_types.append("tables")
            if meta.get("has_formulas"):
                content_types.append("formulas")
            if meta.get("has_code"):
                content_types.append("code")
            if meta.get("has_lists"):
                content_types.append("lists")
            content_str = ", ".join(content_types) if content_types else "none"
            logger.info(f"  Content types: {content_str}")
        
        logger.info("-" * 80)
        
        logger.info(f"\n✅ Metadata inspection complete!")
        
    except Exception as e:
        logger.error(f"Error inspecting metadata: {e}", exc_info=True)


if __name__ == "__main__":
    inspect_chroma_metadata()
