"""Tests for the OpenWebUIFiles class."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from openwebui_client.files import OpenWebUIFiles


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    client = MagicMock()
    # Set up base_url and api_key to be proper strings
    client.base_url = "https://test.com"
    client.api_key = "test_api_key"
    return client


def test_create_validation(mock_client):
    """Test validation when both file and files are provided."""
    files = OpenWebUIFiles(client=mock_client)

    # Mock file data
    file_data = Path(__file__).parent / "data" / "Les_Processus_Cl_s.pdf"
    file_metadata = {"purpose": "assistants"}

    # Both file and files provided should raise ValueError
    with pytest.raises(ValueError):
        files.create(
            file=file_data,
            file_metadata=file_metadata,
            files=[(file_data, file_metadata)],
        )
