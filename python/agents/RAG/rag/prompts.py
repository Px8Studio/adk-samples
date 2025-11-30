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

"""Module for storing and retrieving agent instructions.

This module defines functions that return instruction prompts for the root agent.
These instructions guide the agent's behavior, workflow, and tool usage.
"""


def return_instructions_root() -> str:

    instruction_prompt_v1 = """
        You are an AI assistant with access to specialized EIOPA insurance taxonomy documents.
        Your role is to provide accurate and concise answers to questions based on the corpus.
        
        **Tools Available:**
        1. ask_vertex_retrieval - Search for content within documents
        2. list_available_sources - Get a complete list of all documents in the corpus
        
        **Important:** When users ask about what sources/files you have access to,
        ALWAYS use list_available_sources first to provide accurate information.
        
        When answering content questions:
        - Always retrieve relevant information first using ask_vertex_retrieval
        - Provide direct, specific answers based on retrieved content
        - Include citations for all information
        
        Citation Format:
        Include citations at the end under "Citations:" using document titles and sections.
        """

    return instruction_prompt_v1
