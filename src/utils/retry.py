"""Shared classification for failures that are safe to retry."""

import httpx
from telegram.error import NetworkError, RetryAfter, TimedOut


class RetryableJobError(Exception):
    """A transient failure that should return a durable job to the queue."""


class TerminalJobError(Exception):
    """A failure that must never replay the enclosing durable update."""


def retry_after_seconds(error: BaseException) -> float | None:
    """Return Telegram's requested backoff, when one is present."""
    for candidate in _exception_chain(error):
        if isinstance(candidate, RetryAfter):
            return float(candidate.retry_after)
    return None


def is_retryable_error(error: BaseException) -> bool:
    """Whether an exception represents a transient transport/server failure."""
    chain = _exception_chain(error)
    if any(isinstance(candidate, TerminalJobError) for candidate in chain):
        return False
    for candidate in chain:
        if getattr(candidate, "retryable", False):
            return True
        if isinstance(candidate, (RetryableJobError, RetryAfter, TimedOut, NetworkError)):
            return True
        if isinstance(candidate, (httpx.TimeoutException, httpx.NetworkError)):
            return True
        if isinstance(candidate, httpx.HTTPStatusError):
            return candidate.response.status_code == 429 or candidate.response.status_code >= 500
        # DNS resolution failures are surfaced as OSError on some platforms.
        if isinstance(candidate, OSError):
            return True
    return False


def _exception_chain(error: BaseException) -> list[BaseException]:
    """Follow chained exceptions without looping forever."""
    result: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        result.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return result
