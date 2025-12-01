import os
import vertexai
from vertexai.preview import rag
from dotenv import load_dotenv

load_dotenv()

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
location = os.environ.get("GOOGLE_CLOUD_LOCATION")

print(f"Project: {project_id}")
print(f"Location: {location}")
print(f"\nInitializing Vertex AI...")
vertexai.init(project=project_id, location=location)

print(f"\nListing all corpora in {location}:")
corpora = list(rag.list_corpora())

for corpus in corpora:
    print(f"\nName: {corpus.display_name}")
    print(f"Full resource name: {corpus.name}")
    print(f"Description: {corpus.description}")
    print(f"Created: {corpus.create_time}")
