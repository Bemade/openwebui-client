"""Tests for the OpenWebUIFiles class."""

import pytest
from openwebui_client.files import OpenWebUIFiles


def test_create_validation(mock_client):
    """Test validation when both file and files are provided."""
    files = OpenWebUIFiles(client=mock_client)

    # Mock file data
    file_data = b"test file content"
    file_metadata = {"purpose": "assistants"}

    # Both file and files provided should raise ValueError
    with pytest.raises(ValueError):
        files.create(
            file=file_data,
            file_metadata=file_metadata,
            files=[(file_data, file_metadata)],
        )
