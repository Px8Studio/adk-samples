# RAG Tools Overview

## Does the Agent Know About These Tools?

**No.** By default, the agent **does not** know about all the tools available in the `vertexai.preview.rag` library.

An AI agent only knows about the tools that are explicitly:
1.  **Defined** as Python functions (or `Tool` objects).
2.  **Added** to the `tools` list in the `Agent` definition.
3.  **Described** in the system prompt (which happens automatically when you add them to the `tools` list).

Currently, your agent only has access to:
1.  `retrieve_rag_documentation` (Wraps `vertexai.preview.rag.retrieval_query`)
2.  `list_available_sources` (Wraps `vertexai.preview.rag.list_files`)

## How Are We Accessing Them?

We are accessing these tools via the **Google Cloud Vertex AI Python SDK** (`vertexai`).

*   **Not MCP:** We are not using the Model Context Protocol (MCP) here. MCP is a standard for connecting agents to external systems, but in this case, we are using the direct Python library provided by Google.
*   **Direct SDK Calls:** When we define a tool like `list_available_sources`, we are writing a standard Python function that imports `vertexai.preview.rag` and calls its functions directly.

## Available Native Tools (Complete List)

The `vertexai.preview.rag` library provides the following functions that can be wrapped as tools.

### Corpus Management
*   `create_corpus`: Create a new knowledge base.
*   `delete_corpus`: Remove a knowledge base.
*   `list_corpora`: See all available knowledge bases.
*   `get_corpus`: Get details of a specific corpus.
*   `update_corpus`: Update metadata (display name, description) of a corpus.

### File Management
*   `upload_file`: Add a document to the corpus (synchronous).
*   `import_files`: Batch add documents from Google Cloud Storage or Google Drive (synchronous).
*   `import_files_async`: Batch add documents asynchronously (returns an Operation).
*   `delete_file`: Remove a document.
*   `get_file`: Get metadata about a document.
*   `list_files`: List files in a corpus (Wrapped by `list_available_sources`).

### RAG Engine Management
*   `get_rag_engine_config`: Retrieve the configuration of the RAG Engine.
*   `update_rag_engine_config`: Update the configuration of the RAG Engine.

### Retrieval
*   `retrieval_query`: Search the corpus (Wrapped by `retrieve_rag_documentation`).

## Configuration Classes

The library also includes many configuration classes used as arguments for the tools above. These are **not** tools themselves, but data structures.

*   **Data Structures:** `RagCorpus`, `RagFile`, `RagResource`
*   **Configurations:** `ChunkingConfig`, `EmbeddingModelConfig`, `RagRetrievalConfig`, `TransformationConfig`
*   **Search Configs:** `VertexAiSearchConfig`, `VertexVectorSearch`, `HybridSearch`, `Ranking`
*   **Sources:** `JiraSource`, `SharePointSource`, `SlackChannelsSource`

## Should We Add Them?

It depends on the **purpose** of your agent:

*   **Consumer/Q&A Agent:** If the agent is meant for end-users to just *ask questions*, you **should not** give it management tools. You don't want a user to accidentally delete your corpus!
*   **Admin/Librarian Agent:** If the agent is meant to help you *manage* your knowledge base, then yes, adding `upload_file`, `delete_file`, and `update_corpus` would be very powerful.

For now, your agent is configured as a **Q&A Agent**, so restricting it to just "Read" capabilities (Retrieval + Listing) is the secure and correct design.

## Potential Enhancements for Q&A

While the current setup is sufficient for most needs, there are two "Read" tools that could enhance a Q&A agent in specific scenarios:

1.  **`get_file(name)`**:
    *   **Use Case:** If users often ask about document metadata, like *"When was the 'Safety Manual' updated?"* or *"Who is the author of this file?"*.
    *   **Value:** Allows the agent to answer questions about the *documents themselves*, not just their content.

2.  **`list_corpora()`**:
    *   **Use Case:** If you have multiple knowledge bases (e.g., "HR Policies", "Technical Docs", "Sales Material") and want the agent to switch between them or know what categories of information are available.
    *   **Value:** Enables a multi-domain Q&A agent.

**Recommendation:** Stick with the current setup unless you specifically need metadata answers or multi-corpus support. Simpler agents are more reliable.
