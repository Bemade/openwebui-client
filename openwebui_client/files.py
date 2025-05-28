"""OpenWebUI files class for handling file uploads."""

import logging
from typing import Dict, Any, Tuple, Optional, Iterable, List, overload

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
    ) -> FileObject:
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

        # OpenWebUI keeps the /v1/files endpoint compatible with OpenAI
        response = self._client.post(
            "/v1/files",
            file=file,
            file_metadata=file_metadata,
            cast_to=FileObject,
        )
        return response
