import os
import vertexai
from vertexai.preview import rag
from dotenv import load_dotenv

load_dotenv()

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
location = os.environ.get("GOOGLE_CLOUD_LOCATION")
rag_corpus = os.environ.get("RAG_CORPUS")

print(f"Project: {project_id}")
print(f"Location: {location}")
print(f"Corpus: {rag_corpus}")

if project_id and location:
    vertexai.init(project=project_id, location=location)

try:
    print("Listing files...")
    # Try with a large page size if supported, or just default
    files_iter = rag.list_files(corpus_name=rag_corpus, page_size=100)
    files = list(files_iter)

    print(f"Total files found: {len(files)}")
    for f in files:
        print(f" - {f.display_name}")

except Exception as e:
    print(f"Error: {e}")
