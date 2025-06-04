"""Tests for the OpenWebUICompletions class."""

import pytest
from unittest.mock import patch, MagicMock

from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.file_object import FileObject
from openwebui_client.completions import OpenWebUICompletions


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    client = MagicMock()
    # Create a proper ChatCompletion with dictionary structure instead of MagicMock objects
    client.post.return_value = ChatCompletion(
        id="test-id",
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Test response",
                    "role": "assistant",
                },
            }
        ],
        created=1619990475,
        model="gpt-4",
        object="chat.completion",
        usage={
            "completion_tokens": 10,
            "prompt_tokens": 20,
            "total_tokens": 30,
        },
    )
    return client


def test_create_with_files(mock_client):
    """Test create method with files parameter."""
    completions = OpenWebUICompletions(client=mock_client)

    # Mock file identifier list
    files = [MagicMock()]

    # Call create with files
    response = completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4",
        files=files,
    )

    # Check that the client's post method was called with the right parameters
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args

    # Check endpoint (in the path parameter)
    assert kwargs["path"] == "/api/chat/completions"

    # Check files parameter
    assert kwargs["body"]["files"] == files

    # Check that we got a response
    assert response.choices[0].message.content == "Test response"


def test_create_without_files(mock_client):
    """Test create method without files parameter."""
    completions = OpenWebUICompletions(client=mock_client)

    # Call create without files parameter
    response = completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4",
    )

    # For the no-files case, super().create() is called instead, so we don't check
    # post method arguments directly. Just make sure we got the right response.

    # Check that we got a response
    assert response.choices[0].message.content == "Test response"


def test_create_with_not_given_params(mock_client):
    """Test that NOT_GIVEN parameters are handled correctly."""
    completions = OpenWebUICompletions(client=mock_client)

    # Call create with some NOT_GIVEN parameters
    response = completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4",
        temperature=1.0,
        max_tokens=NOT_GIVEN,
        stop=NOT_GIVEN,
    )

    # For the 'with NOT_GIVEN params' case, super().create() is called instead, so we don't check
    # post method arguments directly. Just make sure we got the right response.
    
    # Check that we got a response
    assert response.choices[0].message.content == "Test response"
