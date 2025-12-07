import os
import sys
import logging
from dotenv import load_dotenv

# Add the agent directory to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.shared_libraries import local_rag_tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_local_retrieval():
    logger.info("Starting local retrieval test...")

    # Check if CHROMA_DB_PATH is set
    chroma_path = os.environ.get("CHROMA_DB_PATH")
    if not chroma_path:
        # Fallback to the user's specified path for this test session if not in env
        # User said: C:\Users\rjjaf\_Projects\solven\backend\data\chroma_db
        default_path = r"C:\Users\rjjaf\_Projects\solven\backend\data\chroma_db"
        logger.info(f"CHROMA_DB_PATH not set, using default for test: {default_path}")
        os.environ["CHROMA_DB_PATH"] = default_path

    # Check if the DB path exists
    if not os.path.exists(os.environ["CHROMA_DB_PATH"]):
        logger.error(f"ChromaDB path does not exist: {os.environ['CHROMA_DB_PATH']}")
        # We can't proceed if the DB isn't there, but we can try to proceed if the user hasn't run ingestion yet.
        # But this test expects to retrieve something.
        return

    # Test listing sources
    logger.info("Testing list_chroma_sources...")
    sources = local_rag_tool.list_chroma_sources()
    print(f"Sources: {sources}")

    # Test retrieval
    query = "What is EIOPA?"
    logger.info(f"Testing retrieval for query: '{query}'...")
    result = local_rag_tool.retrieve_chroma_documentation(query)

    print("\n--- Retrieval Result ---")
    print(result)
    print("------------------------\n")

    if "Error" in result:
        logger.error("Test FAILED: Retrieval returned an error.")
    elif "No relevant documentation" in result:
        logger.warning("Test INCONCLUSIVE: No docs found (maybe empty DB?).")
    else:
        logger.info("Test PASSED: Successfully retrieved documentation.")

    # Test metadata retrieval (Pick the first file from sources if available)
    if sources and not sources[0].startswith("Collection contains"):
        test_file = sources[0]
        logger.info(f"Testing get_chroma_file_metadata for '{test_file}'...")
        metadata = local_rag_tool.get_chroma_file_metadata(test_file)
        print(f"Metadata: {metadata}")
        if "No metadata found" in metadata or "Error" in metadata:
            logger.warning("Metadata test inconclusive/failed.")
        else:
            logger.info("Metadata test PASSED.")


if __name__ == "__main__":
    load_dotenv()
    test_local_retrieval()
