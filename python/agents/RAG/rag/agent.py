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
RAG Agent - Local Mode (ChromaDB + Google AI API)

This agent uses a local ChromaDB vector store for document retrieval.
It requires a Gemini model that supports function calling.

Features:
- Local ChromaDB for document storage and retrieval
- Response metadata tracking (tokens, costs, timing)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent

from .shared_libraries import local_rag_tool
from .plugins import ResponseMetadataPlugin

load_dotenv()

# --- Plugin Configuration ---
# Path to rate limits config (optional, enhances metadata display)
LIMITS_CONFIG_PATH = os.environ.get(
    "LIMITS_CONFIG_PATH",
    str(
        Path(__file__).parent.parent.parent.parent.parent
        / "solven"
        / "config"
        / "google_paid_tier_limits.yaml"
    ),
)

# Initialize the metadata plugin
response_metadata_plugin = ResponseMetadataPlugin(
    limits_config_path=LIMITS_CONFIG_PATH
    if Path(LIMITS_CONFIG_PATH).exists()
    else None,
    show_metadata=os.environ.get("SHOW_RESPONSE_METADATA", "true").lower() == "true",
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Configuration ---
# Model must support function calling. Gemma models do NOT.
# Recommended: gemini-2.0-flash (1500 RPD free tier), gemini-2.5-flash (20 RPD)
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash")

logger.info(f"RAG Agent using model: {MODEL_NAME}")


# --- Agent Instruction ---
INSTRUCTION = """
You are a helpful AI assistant with access to a local document knowledge base.
Your role is to provide accurate answers based ONLY on the documents you can retrieve.

**Your Workflow:**
1.  **Understand the Question:** Analyze what the user is asking.
2.  **Search for Information:** Use the `retrieve_chroma_documentation` tool to find relevant content.
3.  **Synthesize the Answer:** Combine the retrieved information into a clear response.
4.  **Cite Your Sources:** Include document names and relevant details in your answer.

**Tools Available:**
-   `retrieve_chroma_documentation(query)`: Search the knowledge base for relevant documents.
-   `list_chroma_sources()`: List all available documents in the knowledge base.
-   `get_chroma_file_metadata(file_name)`: Get details about a specific document.

**Important Rules:**
-   NEVER answer from your own knowledge. Only use retrieved documents.
-   If no relevant information is found, say so clearly.
-   When listing sources, show the COMPLETE list from the tool, don't summarize.
"""


# --- Define the Agent ---
root_agent = Agent(
    model=MODEL_NAME,
    name="ask_rag_agent",
    instruction=INSTRUCTION,
    tools=[
        local_rag_tool.retrieve_chroma_documentation,
        local_rag_tool.list_chroma_sources,
        local_rag_tool.get_chroma_file_metadata,
    ],
)

# Set the model name in the plugin for accurate cost estimation
# (The plugin also gets this from llm_response.model_version when available)
response_metadata_plugin._model_used = MODEL_NAME

logger.info("RAG Agent initialized with local ChromaDB tools.")
logger.info(
    f"Response metadata plugin enabled: {response_metadata_plugin._show_metadata}"
)

# Export for `--extra_plugins` CLI option:
# Use: uv run adk web --extra_plugins rag.agent.response_metadata_plugin
