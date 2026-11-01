"""ASGI middleware for request size limiting and DoS mitigation."""
import json
from typing import Optional
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send


class RequestSizeLimitMiddleware:
    """Enforces maximum HTTP request body size at the ASGI boundary.

    Protects against denial-of-service and memory/disk exhaustion:
    1. Fast rejection: Rejects immediately if Content-Length header exceeds max_size_bytes.
    2. Streaming metering: Wraps the ASGI receive callable to count raw incoming body bytes,
       aborting with HTTP 413 if the payload exceeds max_size_bytes, even if Content-Length
       is missing, chunked, or forged.
    """

    def __init__(self, app: ASGIApp, max_size_bytes: Optional[int] = None):
        self.app = app
        self._max_size_bytes = max_size_bytes

    def get_max_size(self) -> int:
        if self._max_size_bytes is not None:
            return self._max_size_bytes
        from app.core.config import settings
        return settings.MAX_REQUEST_BODY_SIZE_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_limit = self.get_max_size()
        headers = Headers(scope=scope)
        content_length_header = headers.get("content-length")

        if content_length_header is not None:
            try:
                content_length = int(content_length_header)
                if content_length > max_limit:
                    response = Response(
                        content=json.dumps({
                            "detail": f"Request payload exceeds maximum allowed limit of {max_limit} bytes."
                        }),
                        status_code=413,
                        media_type="application/json",
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        bytes_received = 0

        async def limited_receive():
            nonlocal bytes_received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                bytes_received += len(body)
                if bytes_received > max_limit:
                    raise StarletteHTTPException(
                        status_code=413,
                        detail=f"Request payload exceeds maximum allowed limit of {max_limit} bytes.",
                    )
            return message

        await self.app(scope, limited_receive, send)
