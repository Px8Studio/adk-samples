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

import logging
import os

import vertexai
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

from .prompts import return_instructions_root

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_location_from_corpus(corpus_name: str) -> str | None:
    """Extract location from corpus resource name.

    Expected format: projects/{project}/locations/{location}/ragCorpora/{id}
    """
    try:
        parts = corpus_name.split("/")
        if len(parts) >= 4 and parts[2] == "locations":
            return parts[3]
    except Exception:
        pass
    return None


# Initialize Vertex AI globally
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
location = os.environ.get("GOOGLE_CLOUD_LOCATION")
rag_corpus = os.environ.get("RAG_CORPUS")

# If RAG_CORPUS is set, extract and use its actual location to avoid mismatch errors
if rag_corpus:
    corpus_location = _extract_location_from_corpus(rag_corpus)
    if corpus_location:
        logger.info(f"Using location '{corpus_location}' extracted from RAG_CORPUS")
        location = corpus_location
    else:
        logger.warning(f"Could not extract location from RAG_CORPUS: {rag_corpus}")

if project_id and location:
    vertexai.init(project=project_id, location=location)


def list_available_sources() -> list[str]:
    """Returns the FULL list of available documents in the RAG corpus.

    The agent must present this list to the user without summarization.
    """
    # CRITICAL: Re-initialize vertexai for each call to ensure correct location
    # Extract location from corpus to avoid mismatch errors
    rag_corpus = os.environ.get("RAG_CORPUS")
    if not rag_corpus:
        return ["No RAG corpus configured."]

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = _extract_location_from_corpus(rag_corpus) or os.environ.get(
        "GOOGLE_CLOUD_LOCATION"
    )

    if project_id and location:
        vertexai.init(project=project_id, location=location)

    try:
        files = list(rag.list_files(corpus_name=rag_corpus))
        if not files:
            return ["No files found in corpus."]

        return [f.display_name for f in files]
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        return [f"Error listing sources: {e}"]


def get_file_metadata(file_name: str) -> str:
    """Returns metadata for a specific file in the RAG corpus.

    Args:
        file_name: The display name of the file to get metadata for.

    Returns:
        A formatted string containing the file's metadata.
    """
    # CRITICAL: Re-initialize vertexai for each call to ensure correct location
    rag_corpus = os.environ.get("RAG_CORPUS")
    if not rag_corpus:
        return "No RAG corpus configured."

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = _extract_location_from_corpus(rag_corpus) or os.environ.get(
        "GOOGLE_CLOUD_LOCATION"
    )

    if project_id and location:
        vertexai.init(project=project_id, location=location)

    try:
        files = list(rag.list_files(corpus_name=rag_corpus))
        for f in files:
            if f.display_name == file_name:
                return (
                    f"File: {f.display_name}\n"
                    f"Name: {f.name}\n"
                    f"Created: {f.create_time}\n"
                    f"Updated: {f.update_time}\n"
                    f"Description: {f.description}"
                )
        return f"File '{file_name}' not found in corpus."
    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        return f"Error getting file metadata: {e}"


def list_rag_corpora() -> list[str]:
    """Lists all available RAG corpora in the project."""
    # CRITICAL: Re-initialize vertexai for each call
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

    # Try to extract location from RAG_CORPUS first, fallback to env var
    rag_corpus = os.environ.get("RAG_CORPUS")
    location = (
        _extract_location_from_corpus(rag_corpus) if rag_corpus else None
    ) or os.environ.get("GOOGLE_CLOUD_LOCATION")

    if project_id and location:
        vertexai.init(project=project_id, location=location)

    try:
        corpora = list(rag.list_corpora())
        if not corpora:
            return ["No corpora found."]

        return [f"{c.display_name} ({c.name})" for c in corpora]
    except Exception as e:
        logger.error(f"Error listing corpora: {e}")
        return [f"Error listing corpora: {e}"]


# Build tools list conditionally based on RAG_CORPUS availability
tools = []
rag_corpus = os.environ.get("RAG_CORPUS")

# Configuration
model_name = os.environ.get("MODEL_NAME")
rag_similarity_top_k = int(os.environ.get("RAG_SIMILARITY_TOP_K"))
rag_vector_distance_threshold = float(os.environ.get("RAG_VECTOR_DISTANCE_THRESHOLD"))

if rag_corpus:
    ask_vertex_retrieval = VertexAiRagRetrieval(
        name="retrieve_rag_documentation",
        description=(
            "Use this tool to retrieve documentation and reference materials for the question from the RAG corpus,"
        ),
        rag_resources=[
            rag.RagResource(
                # please fill in your own rag corpus
                # here is a sample rag corpus for testing purpose
                # e.g. projects/123/locations/us-central1/ragCorpora/456
                rag_corpus=rag_corpus
            )
        ],
        similarity_top_k=rag_similarity_top_k,
        vector_distance_threshold=rag_vector_distance_threshold,
    )
    tools.append(ask_vertex_retrieval)
    tools.append(list_available_sources)
    tools.append(get_file_metadata)
    tools.append(list_rag_corpora)
else:
    logger.warning(
        "RAG_CORPUS environment variable not set. RAG capabilities will be disabled."
    )

root_agent = Agent(
    model=model_name,
    name="ask_rag_agent",
    instruction=return_instructions_root(),
    tools=tools,
)
