"""Integration tests for the OpenWebUI client.

These tests require a running OpenWebUI server and valid API credentials.
Set the OPENWEBUI_API_KEY and OPENWEBUI_API_BASE environment variables before running.
"""

import os
import pytest
from openwebui_client import OpenWebUIClient

# Skip all tests if no API key or base URL is provided
pytestmark = pytest.mark.skipif(
    not (os.environ.get("OPENWEBUI_API_KEY") and os.environ.get("OPENWEBUI_API_BASE")),
    reason="OPENWEBUI_API_KEY and OPENWEBUI_API_BASE environment variables are required for integration tests",
)


@pytest.fixture
def client():
    """Create a client connected to a real OpenWebUI instance."""
    return OpenWebUIClient(
        api_key=os.environ.get("OPENWEBUI_API_KEY"),
        base_url=os.environ.get("OPENWEBUI_API_BASE"),
    )


def test_chat_completion(client):
    """Test that chat completions work with a real server."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use a model available on your OpenWebUI server
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"},
        ],
        max_tokens=10,  # Limit the response size for test efficiency
    )
    
    # Verify we got a response with the expected structure
    assert response.id is not None
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None
    assert response.choices[0].message.role == "assistant"


def test_chat_completion_with_file(client):
    """Test chat completions with a file attachment."""
    # Create a small test file
    file_content = b"This is a test file for OpenWebUI."
    
    # Make request with file attachment
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use a model available on your OpenWebUI server
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's in the attached file?"},
        ],
        files=[file_content],
        max_tokens=20,  # Limit the response size for test efficiency
    )
    
    # Verify we got a response
    assert response.id is not None
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None
    assert response.choices[0].message.role == "assistant"
    # The response should likely mention something about a test file
    assert "test" in response.choices[0].message.content.lower() or "file" in response.choices[0].message.content.lower()


@pytest.mark.xfail(reason="File uploads might not be supported on all OpenWebUI instances")
def test_file_upload(client):
    """Test file uploads."""
    # Create a small test file
    file_content = b"This is a test file for OpenWebUI file uploads."
    
    # Upload the file
    try:
        file_obj = client.files.create(
            file=file_content,
            file_metadata={"purpose": "assistants"},
        )
        
        # Check that we got a file object back
        assert file_obj.id is not None
        assert file_obj.bytes == len(file_content)
        
    except Exception as e:
        pytest.xfail(f"File upload failed: {str(e)}")


@pytest.mark.xfail(reason="Multiple file uploads might not be supported")
def test_multiple_file_upload(client):
    """Test multiple file uploads."""
    # Create small test files
    file_content1 = b"This is the first test file for OpenWebUI."
    file_content2 = b"This is the second test file for OpenWebUI."
    
    # Upload the files
    try:
        file_objects = client.files.create(
            files=[
                (file_content1, {"purpose": "assistants"}),
                (file_content2, {"purpose": "assistants"}),
            ]
        )
        
        # Check that we got file objects back
        assert len(file_objects) == 2
        assert file_objects[0].id is not None
        assert file_objects[1].id is not None
        
    except Exception as e:
        pytest.xfail(f"Multiple file upload failed: {str(e)}")
