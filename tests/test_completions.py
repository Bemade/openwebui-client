"""Tests for the OpenWebUICompletions class."""

import pytest
from unittest.mock import patch, MagicMock

from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openwebui_client.completions import OpenWebUICompletions


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    client = MagicMock()
    client.post.return_value = ChatCompletion(
        id="test-id",
        choices=[
            MagicMock(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Test response",
                    role="assistant",
                ),
            )
        ],
        created=1619990475,
        model="gpt-4",
        object="chat.completion",
        usage=MagicMock(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
        ),
    )
    return client


def test_create_with_files(mock_client):
    """Test create method with files parameter."""
    completions = OpenWebUICompletions(client=mock_client)

    # Mock file data
    file_data = b"test file content"

    # Call create with files
    response = completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4",
        files=[file_data],
    )

    # Check that the client's post method was called with the right parameters
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args

    # Check endpoint
    assert args[0] == "/openai/chat/completions"

    # Check files parameter
    assert kwargs["files"] == [file_data]

    # Check that we got a response
    assert response.choices[0].message.content == "Test response"


def test_create_without_files(mock_client):
    """Test create method without files parameter."""
    completions = OpenWebUICompletions(client=mock_client)

    # Set up a spy on super().create
    with patch.object(OpenWebUICompletions, "__bases__",
                      spec=OpenWebUICompletions.__bases__) as mock_base:
        # Set up the mock for super().create
        mock_super = MagicMock()
        mock_base[0].create.return_value = mock_client.post.return_value

        # Call create without files
        response = completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
        )

    # Check that we got a response
    assert response.choices[0].message.content == "Test response"


def test_create_with_not_given_params(mock_client):
    """Test that NOT_GIVEN parameters are handled correctly."""
    completions = OpenWebUICompletions(client=mock_client)

    # Call create with some NOT_GIVEN parameters
    completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4",
        temperature=1.0,
        max_tokens=NOT_GIVEN,
        stop=NOT_GIVEN,
    )

    # Check that NOT_GIVEN parameters weren't passed to the API
    _, kwargs = mock_client.post.call_args
    all_params = kwargs

    # These should be included
    assert "messages" in all_params
    assert "model" in all_params
    assert "temperature" in all_params

    # These should be filtered out
    assert "max_tokens" not in all_params
    assert "stop" not in all_params
