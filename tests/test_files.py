"""Tests for the OpenWebUIFiles class."""

import pytest
from unittest.mock import patch, MagicMock

from openai.types import FileObject
from openwebui_client.files import OpenWebUIFiles


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    client = MagicMock()
    client.post.return_value = FileObject(
        id="file-123",
        bytes=100,
        created_at=1619990475,
        filename="test.txt",
        object="file",
        purpose="assistants",
        status="processed",
        status_details=None,
    )
    return client


def test_create_single_file(mock_client):
    """Test creating a single file."""
    files = OpenWebUIFiles(client=mock_client)
    
    # Mock file data
    file_data = b"test file content"
    file_metadata = {"purpose": "assistants"}
    
    # Call create with a single file
    response = files.create(
        file=file_data,
        file_metadata=file_metadata,
    )
    
    # Check that the client's post method was called with the right parameters
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    
    # Check endpoint
    assert args[0] == "/v1/files"
    
    # Check file parameter
    assert kwargs["file"] == file_data
    assert kwargs["file_metadata"] == file_metadata
    
    # Check that we got a response
    assert response.id == "file-123"
    assert response.purpose == "assistants"


def test_create_multiple_files(mock_client):
    """Test creating multiple files."""
    files = OpenWebUIFiles(client=mock_client)
    
    # Mock file data
    file_data1 = b"test file content 1"
    file_data2 = b"test file content 2"
    file_metadata1 = {"purpose": "assistants"}
    file_metadata2 = {"purpose": "fine-tune"}
    
    # Set up the create method to be called multiple times
    files.create = MagicMock(side_effect=[
        FileObject(
            id="file-123",
            bytes=len(file_data1),
            created_at=1619990475,
            filename="test1.txt",
            object="file",
            purpose="assistants",
            status="processed",
            status_details=None,
        ),
        FileObject(
            id="file-456",
            bytes=len(file_data2),
            created_at=1619990476,
            filename="test2.txt",
            object="file",
            purpose="fine-tune",
            status="processed",
            status_details=None,
        ),
    ])
    
    # Original method reference for later restoration
    original_create = OpenWebUIFiles.create
    
    try:
        # Call create with multiple files
        response = original_create(
            files,
            files=[
                (file_data1, file_metadata1),
                (file_data2, file_metadata2),
            ],
        )
        
        # Check that create was called twice
        assert files.create.call_count == 2
        
        # Check first call
        args, kwargs = files.create.call_args_list[0]
        assert kwargs["file"] == file_data1
        assert kwargs["file_metadata"] == file_metadata1
        
        # Check second call
        args, kwargs = files.create.call_args_list[1]
        assert kwargs["file"] == file_data2
        assert kwargs["file_metadata"] == file_metadata2
        
        # Check that we got a list of responses
        assert len(response) == 2
        assert response[0].id == "file-123"
        assert response[1].id == "file-456"
    finally:
        # Restore original method to avoid affecting other tests
        files.create = original_create


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
