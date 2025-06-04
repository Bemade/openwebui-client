"""OpenWebUI files class for handling file uploads."""

import logging
from typing import Dict, Any, Tuple, Optional, Iterable, List, overload, Union
import time

from openai import NOT_GIVEN
from openai.resources.files import Files, FileObject

from pathlib import Path

_logger = logging.getLogger(__name__)


class OpenWebUIFiles(Files):
    """Extended Files class for OpenWebUI with improved file upload functionality."""

    @overload
    def create(
        self,
        files: Iterable[Tuple[Path, Optional[Dict[str, Any]]]],
    ) -> List[FileObject]: ...

    @overload
    def create(
        self,
        file: Path,
        file_metadata: Optional[Dict[str, Any]],
    ) -> FileObject: ...

    def create(
        self,
        file: Path = None,
        file_metadata: Optional[Dict[str, Any]] = None,
        files: Optional[Iterable[Tuple[bytes, Optional[Dict[str, Any]]]]] = None,
    ) -> Union[FileObject, List[FileObject]]:
        """Upload a file to the OpenWebUI API.

        Args:
            file: The file content as bytes
            file_metadata: Additional metadata for the file
            files: Multiple files to upload at once

        Returns:
            FileObject or List[FileObject]: The uploaded file object(s)

        Raises:
            ValueError: If both file and files are provided
        """
        if file and files:
            raise ValueError("file and files cannot both be specified")
        elif files:
            return [self.create(file=f, file_metadata=meta) for f, meta in files]

        with file.open("rb") as filestream:
            # OpenWebUI requires a specific format for file uploads
            # The key differences from standard OpenAI:
            # 1. Using a trailing slash on the endpoint path
            # 2. Adding a 'process=true' parameter
            # 3. Using the proper multipart/form-data format for the file

            # Use direct HTTP request instead of the OpenAI client for file uploads
            import requests

            # Extract the base URL from the client (removing any trailing slash)
            base_url = str(self._client.base_url).rstrip('/')

            # Construct the full URL with the required trailing slash
            url = f"{base_url}/v1/files/"

            # Set up authentication headers
            headers = {
                "Authorization": f"Bearer {self._client.api_key}"
            }

            # Set up the multipart form data like the curl command
            # Place both the file and process=true in the files parameter
            # This matches how curl -F works
            files = {
                "file": filestream,
            }
            data = {
                "process": "true"
            }

            # Add any additional metadata provided by the user
            if file_metadata:
                for key, value in file_metadata.items():
                    files[key] = (None, str(value))

            # Print detailed request information
            _logger.debug(f"FILES API - URL: {url}")
            _logger.debug(f"FILES API - Headers: {headers}")
            _logger.debug(f"FILES API - Data: {data}")
            _logger.debug(f"FILES API - Files: {files.keys()}")

            # Make the HTTP request directly
            http_response = requests.post(url, headers=headers, files=files, data=data)

            # Print response details
            _logger.debug(f"FILES API - Response Status: {http_response.status_code}")
            _logger.debug(f"FILES API - Response Headers: {dict(http_response.headers)}")
            _logger.debug(f"FILES API - Response Body: {http_response.text[:500]}..." if len(http_response.text) > 500 else f"FILES API - Response Body: {http_response.text}")

            # Raise an exception for any HTTP error
            http_response.raise_for_status()

            # Parse the JSON response
            response_data = http_response.json()

            if response_data.get("error"):
                raise ValueError(response_data.get("error"))

            # Convert the response to an OpenAI FileObject with required defaults
            file_object = FileObject(
                id=response_data.get("id", f"file-{str(file.name)}"),
                bytes=response_data.get("bytes", file.stat().st_size),  # Default to file size
                created_at=response_data.get("created_at", int(time.time())),  # Default to current time
                filename=response_data.get("filename", file.name),
                object="file",  # Required fixed value
                purpose=response_data.get("purpose", "assistants"),  # Default purpose
                status=response_data.get("status", "processed"),  # Default status
                status_details=response_data.get("status_details"),
            )

            return file_object