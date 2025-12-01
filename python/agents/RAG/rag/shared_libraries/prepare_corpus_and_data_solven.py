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

from google.auth import default
from google.api_core.exceptions import ResourceExhausted
import vertexai
from vertexai.preview import rag
import os
from dotenv import load_dotenv, set_key

# Load environment variables from .env file
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError(
        "GOOGLE_CLOUD_PROJECT environment variable not set. Please set it in your .env file."
    )
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
if not LOCATION:
    raise ValueError(
        "GOOGLE_CLOUD_LOCATION environment variable not set. Please set it in your .env file."
    )
CORPUS_DISPLAY_NAME = "EIOPA_Insurance_Taxonomy_Corpus"
CORPUS_DESCRIPTION = "Corpus containing EIOPA insurance taxonomy documentation"
PDF_DIRECTORY = r"C:\Users\rjjaf\_Projects\solven\backend\data\eiopa\insurance\input"
if not os.path.isdir(PDF_DIRECTORY):
    print(f"PDF directory '{PDF_DIRECTORY}' does not exist; creating it.")
    os.makedirs(PDF_DIRECTORY, exist_ok=True)

ENV_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".env")
)


def initialize_vertex_ai():
    credentials, _ = default()
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)


def create_or_get_corpus():
    embedding_model_config = rag.EmbeddingModelConfig(
        publisher_model="publishers/google/models/text-embedding-004"
    )
    existing_corpora = rag.list_corpora()
    corpus = None
    for existing_corpus in existing_corpora:
        if existing_corpus.display_name == CORPUS_DISPLAY_NAME:
            corpus = existing_corpus
            print(f"Found existing corpus with display name '{CORPUS_DISPLAY_NAME}'")
            break
    if corpus is None:
        # IMPORTANT: Add backend_config to ensure corpus is created in the specified LOCATION
        # Without this, the corpus defaults to 'global' location regardless of vertexai.init()
        backend_config = rag.RagVectorDbConfig(
            rag_managed_db=rag.RagManagedDb(),
        )
        corpus = rag.create_corpus(
            display_name=CORPUS_DISPLAY_NAME,
            description=CORPUS_DESCRIPTION,
            embedding_model_config=embedding_model_config,
            backend_config=backend_config,
        )
        print(f"Created new corpus '{CORPUS_DISPLAY_NAME}' in location: {LOCATION}")
        print(f"Corpus name: {corpus.name}")
    return corpus


def list_corpus_files(corpus_name):
    files = list(rag.list_files(corpus_name=corpus_name))
    print(f"Total files in corpus: {len(files)}")
    for file in files:
        print(f"File: {file.display_name} - {file.name}")
    return files


def delete_all_corpus_files(corpus_name):
    files = list(rag.list_files(corpus_name=corpus_name))
    for file in files:
        print(f"Deleting file: {file.display_name} - {file.name}")
        rag.delete_file(name=file.name)
    print("All files deleted from corpus.")


def upload_pdf_to_corpus(corpus_name, pdf_path, display_name, description):
    print(f"Uploading {display_name} to corpus...")
    try:
        rag_file = rag.upload_file(
            corpus_name=corpus_name,
            path=pdf_path,
            display_name=display_name,
            description=description,
        )
        print(f"Successfully uploaded {display_name} to corpus")
        return rag_file
    except ResourceExhausted as e:
        print(f"Error uploading file {display_name}: {e}")
        print(
            "\nThis error suggests that you have exceeded the API quota for the embedding model."
        )
        print("This is common for new Google Cloud projects.")
        print(
            "Please see the 'Troubleshooting' section in the README.md for instructions on how to request a quota increase."
        )
        return None
    except Exception as e:
        print(f"Error uploading file {display_name}: {e}")
        return None


def update_env_file(corpus_name, env_file_path):
    set_key(env_file_path, "RAG_CORPUS", corpus_name)
    print(f"Updated RAG_CORPUS in {env_file_path} to {corpus_name}")


def main(cleanup=False):
    initialize_vertex_ai()
    corpus = create_or_get_corpus()
    update_env_file(corpus.name, ENV_FILE_PATH)

    # Optionally clean up all files in the corpus
    if cleanup:
        delete_all_corpus_files(corpus.name)

    # Get existing file display names to avoid duplicates
    existing_files = list_corpus_files(corpus.name)
    existing_display_names = set(file.display_name for file in existing_files)

    # Upload only new PDFs
    for filename in os.listdir(PDF_DIRECTORY):
        if filename.lower().endswith(".pdf"):
            if filename in existing_display_names:
                print(f"Skipping {filename}: already exists in corpus.")
                continue
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            print(f"\nFound PDF: {pdf_path}")
            upload_pdf_to_corpus(
                corpus_name=corpus.name,
                pdf_path=pdf_path,
                display_name=filename,
                description=f"EIOPA document: {filename}",
            )

    # List all files in the corpus
    list_corpus_files(corpus_name=corpus.name)


if __name__ == "__main__":
    # Set cleanup=True to delete all files before uploading
    main(cleanup=True)
