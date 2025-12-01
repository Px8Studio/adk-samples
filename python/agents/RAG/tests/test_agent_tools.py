import os
import pytest
from unittest.mock import patch, MagicMock
from rag.agent import (
    list_available_sources,
    get_file_metadata,
    list_rag_corpora,
    root_agent,
)


def test_list_available_sources_no_corpus():
    with patch.dict(os.environ, {}, clear=True):
        sources = list_available_sources()
        assert sources == ["No RAG corpus configured."]


@patch("rag.agent.rag.list_files")
def test_list_available_sources_with_corpus(mock_list_files):
    with patch.dict(
        os.environ,
        {
            "RAG_CORPUS": "projects/123/locations/us-central1/ragCorpora/456",
            "GOOGLE_CLOUD_PROJECT": "my-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        },
        clear=True,
    ):
        # Mock file objects
        mock_file1 = MagicMock()
        mock_file1.display_name = "doc1.pdf"

        mock_file2 = MagicMock()
        mock_file2.display_name = "doc2.pdf"

        mock_list_files.return_value = [mock_file1, mock_file2]

        sources = list_available_sources()

        assert len(sources) == 2
        assert "doc1.pdf" in sources
        assert "doc2.pdf" in sources
        mock_list_files.assert_called_with(
            corpus_name="projects/123/locations/us-central1/ragCorpora/456"
        )


@patch("rag.agent.rag.list_files")
def test_get_file_metadata(mock_list_files):
    with patch.dict(
        os.environ,
        {"RAG_CORPUS": "projects/123/locations/us-central1/ragCorpora/456"},
        clear=True,
    ):
        mock_file = MagicMock()
        mock_file.display_name = "target.pdf"
        mock_file.name = "projects/123/.../ragFiles/789"
        mock_file.create_time = "2023-01-01"
        mock_file.update_time = "2023-01-02"
        mock_file.description = "Test file"

        mock_list_files.return_value = [mock_file]

        # Test finding the file
        metadata = get_file_metadata("target.pdf")
        assert "target.pdf" in metadata
        assert "2023-01-01" in metadata

        # Test not finding the file
        metadata_missing = get_file_metadata("missing.pdf")
        assert "not found" in metadata_missing


@patch("rag.agent.rag.list_corpora")
def test_list_rag_corpora(mock_list_corpora):
    mock_corpus = MagicMock()
    mock_corpus.display_name = "Test Corpus"
    mock_corpus.name = "projects/123/.../ragCorpora/456"

    mock_list_corpora.return_value = [mock_corpus]

    corpora = list_rag_corpora()
    assert len(corpora) == 1
    assert "Test Corpus" in corpora[0]


def test_agent_initialization():
    assert root_agent.name == "ask_rag_agent"
    pass
