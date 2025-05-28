"""OpenWebUI API client library.

This library provides a client for the OpenWebUI API, compatible with
the OpenAI Python SDK but with extensions specific to OpenWebUI.
"""

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from .client import OpenWebUIClient

# Create a default client instance for ease of use
api_key = os.environ.get("OPENWEBUI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
base_url = os.environ.get("OPENWEBUI_API_BASE", "http://localhost:5000/api")

client = OpenWebUIClient(
    api_key=api_key,
    base_url=base_url,
    default_model="gpt-4",  # Default model
)

# Export key classes and functions
__all__ = [
    "OpenWebUIClient",
    "client",
]

try:
    __version__ = version("openwebui-client")
except PackageNotFoundError:
    __version__ = "unknown"
