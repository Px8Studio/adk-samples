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
from .prompts import return_instructions_root

load_dotenv()

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Plugin Configuration ---
# Path to rate limits config (optional, enhances metadata display)
# Look for config in the solven workspace (sibling to adk-samples)
_PROJECTS_DIR = Path(
    __file__
).parent.parent.parent.parent.parent.parent  # Up to _Projects
LIMITS_CONFIG_PATH = os.environ.get(
    "LIMITS_CONFIG_PATH",
    str(_PROJECTS_DIR / "solven" / "config" / "google_paid_tier_limits.yaml"),
)

# Verify the path exists and log for debugging
_limits_path = Path(LIMITS_CONFIG_PATH)
if _limits_path.exists():
    logger.info(f"Found rate limits config at: {LIMITS_CONFIG_PATH}")
else:
    logger.warning(f"Rate limits config not found at: {LIMITS_CONFIG_PATH}")
    logger.info(f"Searched in _Projects dir: {_PROJECTS_DIR}")

# Initialize the metadata plugin
response_metadata_plugin = ResponseMetadataPlugin(
    limits_config_path=LIMITS_CONFIG_PATH if _limits_path.exists() else None,
    show_metadata=os.environ.get("SHOW_RESPONSE_METADATA", "true").lower() == "true",
)

# --- Configuration ---
# Accept an ordered fallback list via ADK_MODEL_LIST. We still hand the Agent
# a single model; fallback sequencing (if desired) must be handled by the caller
# or by retry logic outside this file.
def _resolve_model_name() -> str:
    fallback_list = os.environ.get("ADK_MODEL_LIST", "")
    if fallback_list:
        models = [m.strip() for m in fallback_list.split(",") if m.strip()]
        if models:
            logger.info(
                "RAG Agent using model from ADK_MODEL_LIST (priority first): %s",
                models[0],
            )
            return models[0]

    env_model = os.environ.get("MODEL_NAME")
    if env_model:
        logger.info("RAG Agent using model from MODEL_NAME: %s", env_model)
        return env_model

    default_model = "gemini-2.0-flash"
    logger.info("RAG Agent using default model: %s", default_model)
    return default_model


MODEL_NAME = _resolve_model_name()


# --- Agent Instruction ---
# Instructions are defined in prompts.py for better separation of concerns
INSTRUCTION = return_instructions_root()


# --- Define the Agent ---
root_agent = Agent(
    model=MODEL_NAME,
    name="ask_rag_agent",
    instruction=INSTRUCTION,
    tools=[
        local_rag_tool.retrieve_chroma_documentation,
        local_rag_tool.list_chroma_sources,
        local_rag_tool.get_chroma_file_metadata,
        local_rag_tool.get_corpus_content_summary,
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
