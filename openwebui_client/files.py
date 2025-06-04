"""OpenWebUI files class for handling file uploads."""

import logging
from typing import Dict, Any, Tuple, Optional, Iterable, List, overload, Union

from openai import NOT_GIVEN
from openai.resources.files import Files, FileObject

_logger = logging.getLogger(__name__)


class OpenWebUIFiles(Files):
    """Extended Files class for OpenWebUI with improved file upload functionality."""

    @overload
    def create(
        self,
        files: Iterable[Tuple[bytes, Optional[Dict[str, Any]]]],
    ) -> List[FileObject]: ...

    @overload
    def create(
        self,
        file: bytes,
        file_metadata: Optional[Dict[str, Any]],
    ) -> FileObject: ...

    def create(
        self,
        file: Optional[bytes] = None,
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

        # OpenWebUI requires a specific format for file uploads
        # The key differences from standard OpenAI:
        # 1. Using a trailing slash on the endpoint path
        # 2. Adding a 'process=true' parameter
        # 3. Using the proper multipart/form-data format for the file

        import tempfile
        import os

        # Create a temporary file from the binary data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as temp_file:
            temp_file.write(file)
            temp_path = temp_file.name

        # Use direct HTTP request instead of the OpenAI client for file uploads
        import requests
        
        try:
            # Extract the base URL from the client (removing any trailing slash)
            base_url = str(self._client.base_url).rstrip('/')
            
            # Construct the full URL with the required trailing slash
            url = f"{base_url}/v1/files/"
            
            # Set up authentication headers
            headers = {
                "Authorization": f"Bearer {self._client.api_key}"
            }
            
            # Prepare the data parameters - OpenWebUI requires process=true
            data = {"process": "true"}
            
            # Add any additional metadata provided by the user
            if file_metadata:
                data.update(file_metadata)
            
            # Open the file for reading
            file_handle = open(temp_path, "rb")
            
            # Set up the file parameter with the proper content type
            files = {
                "file": ("file.bin", file_handle, "application/octet-stream")
            }
            
            # Make the HTTP request directly
            http_response = requests.post(url, headers=headers, files=files, data=data)
            
            # Raise an exception for any HTTP error
            http_response.raise_for_status()
            
            # Parse the JSON response
            response_data = http_response.json()
            
            # Convert the response to an OpenAI FileObject
            file_object = FileObject(
                id=response_data.get("id"),
                bytes=response_data.get("bytes"),
                created_at=response_data.get("created_at"),
                filename=response_data.get("filename"),
                object=response_data.get("object"),
                purpose=response_data.get("purpose"),
                status=response_data.get("status"),
                status_details=response_data.get("status_details"),
            )
            
            return file_object
        finally:
            # Close the file handle if it exists
            if 'file_handle' in locals() and file_handle and not file_handle.closed:
                file_handle.close()
            
            # Remove the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return response
