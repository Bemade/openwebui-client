"""OpenWebUI client for interacting with the OpenWebUI API."""

import logging
from typing import Any, Dict, Optional, Union

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types import FileObject

from .completions import OpenWebUICompletions
from .files import OpenWebUIFiles

_logger = logging.getLogger(__name__)


class OpenWebUIClient:
    """Client for interacting with the OpenWebUI API.

    This client is a drop-in replacement for the OpenAI client, with
    extensions for OpenWebUI-specific features like file attachments
    in chat completions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:5000/api",
        default_model: str = "gpt-4",
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenWebUI client.

        Args:
            api_key: Your OpenWebUI API key
            base_url: Base URL for the API (defaults to OpenWebUI's local API)
            default_model: Default model to use for completions
            **kwargs: Additional arguments to pass to the OpenAI client
        """
        # OpenWebUI has different endpoint patterns than OpenAI
        # Don't add /v1 to the base URL as it's inconsistent across endpoints
        # The completions endpoint is at /chat/completions (no /v1)
        # The files endpoint is at /v1/files
        # Remove trailing slash if present
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        
        # Initialize the standard OpenAI client
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
        
        # Replace standard components with OpenWebUI-specific versions
        self.client.chat.completions = OpenWebUICompletions(client=self.client._client)
        self.client.files = OpenWebUIFiles(client=self.client._client)
        
        # Store configuration
        self.base_url = base_url
        self.default_model = default_model
        self.api_key = api_key
    
    @property
    def chat(self) -> Any:
        """Access the chat resources of the client."""
        return self.client.chat
    
    @property
    def files(self) -> OpenWebUIFiles:
        """Access the files resources of the client."""
        return self.client.files
