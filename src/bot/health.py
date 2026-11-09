"""Minimal HTTP health endpoint reporting whether Telegram polling is alive."""

import asyncio
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

HEALTH_HOST = None  # all interfaces: the port is published from the container
HEALTH_PORT = 28120
STALE_AFTER_SECONDS = 90
START_GRACE_SECONDS = 60


def _response(status: int, reason: str, body: str) -> bytes:
    payload = body.encode()
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Connection: close\r\n\r\n"
    )
    return head.encode() + payload


def build_response(
    request_line: bytes, last_success: float | None, started_at: float, now: float
) -> bytes:
    """Build the full HTTP response for a request line.

    Args:
        request_line: First line of the request, e.g. ``GET /health HTTP/1.1``.
        last_success: Monotonic time of the last successful poll, or None.
        started_at: Monotonic time the runner started.
        now: Current monotonic time.

    Returns:
        The raw HTTP response; it carries no ids, tokens or error text.
    """
    parts = request_line.split()
    if len(parts) < 2 or parts[0] != b"GET" or parts[1] != b"/health":
        return _response(404, "Not Found", "not found")
    if last_success is not None:
        alive = now - last_success <= STALE_AFTER_SECONDS
    else:
        alive = now - started_at <= START_GRACE_SECONDS
    if alive:
        return _response(200, "OK", "ok")
    return _response(503, "Service Unavailable", "stale")


async def start_health_server(
    port: int,
    get_state: Callable[[], tuple[float | None, float, float]],
) -> asyncio.Server | None:
    """Start the health server; return None (after one log line) if it cannot bind.

    Args:
        port: TCP port; 0 picks an ephemeral port.
        get_state: Returns (last_success, started_at, now) in monotonic seconds.
    """

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            writer.write(build_response(line, *get_state()))
            await writer.drain()
        except (OSError, TimeoutError, ValueError):
            pass
        finally:
            writer.close()

    try:
        return await asyncio.start_server(handle, HEALTH_HOST, port)
    except OSError:
        logger.error("Health endpoint could not bind its port; polling continues")
        return None
