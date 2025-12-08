import chromadb
import os

CHROMA_DB_PATH = r"C:\Users\rjjaf\_Projects\solven\backend\data\chroma_db"


def main():
    print(f"Checking path: {CHROMA_DB_PATH}")
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Chroma DB path validation failed: {CHROMA_DB_PATH}")
        return

    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        print("Getting collection 'eiopa_insurance_google_004'...")
        collection = client.get_collection("eiopa_insurance_google_004")
        count = collection.count()
        print(f"Collection count: {count}")

        if count > 0:
            # Peek at one item to see metadata
            print("Peeking at one item...")
            result = collection.peek(limit=1)
            print("Sample Metadata:")
            print(result["metadatas"])
        else:
            print("Collection is empty.")

    except Exception as e:
        print(f"Error accessing collection: {e}")


if __name__ == "__main__":
    main()
