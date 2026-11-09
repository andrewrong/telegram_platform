"""Tests for the durable Telegram update inbox."""

from pathlib import Path

import pytest
from telegram import Update
from telegram.error import TimedOut

from src.bot.reliable import ALLOWED_UPDATES, ReliableUpdateRunner, UpdateStore, retry_delay
from src.utils.retry import TerminalJobError, is_retryable_error


def make_update(update_id: int) -> Update:
    return Update.de_json(
        {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "date": 1_700_000_000,
                "chat": {"id": 123, "type": "private"},
                "text": "hello",
            },
        },
        bot=None,
    )


@pytest.mark.asyncio
async def test_ingest_is_deduplicated_and_checkpointed(tmp_path: Path) -> None:
    store = UpdateStore(tmp_path / "inbox.db")
    try:
        update = make_update(42)
        await store.ingest([update, update])

        job = await store.claim()

        assert job is not None
        assert job.update_id == 42
        assert job.payload["message"]["text"] == "hello"
        assert await store.get_offset() == 43
        await store.complete(job.update_id)
        assert await store.claim() is None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_retry_survives_store_reopen(tmp_path: Path) -> None:
    database = tmp_path / "inbox.db"
    store = UpdateStore(database)
    await store.ingest([make_update(7)])
    job = await store.claim()
    assert job is not None
    await store.retry(job, TimedOut())
    store.close()

    reopened = UpdateStore(database)
    try:
        assert await reopened.get_offset() == 8
        # The queued task is preserved even when its next retry is in the future.
        row = reopened._connection.execute(
            "SELECT state, attempts FROM update_jobs WHERE update_id = 7"
        ).fetchone()
        assert row["state"] == "retry"
        assert row["attempts"] == 1
    finally:
        reopened.close()


def test_transient_telegram_error_is_retryable() -> None:
    assert is_retryable_error(TimedOut())


def test_terminal_error_precedes_transient_errors_anywhere_in_chain() -> None:
    """A cloud no-replay marker survives wrappers and underlying network failures."""
    terminal = TerminalJobError("manual resend required")
    terminal.__cause__ = TimedOut()
    wrapper = TimedOut()
    wrapper.__cause__ = terminal
    assert not is_retryable_error(wrapper)


def test_retry_delay_is_bounded() -> None:
    assert 1.6 <= retry_delay(1) <= 2.4
    assert 720 <= retry_delay(100) <= 1080


@pytest.mark.asyncio
async def test_polling_asks_for_every_update_type(tmp_path: Path) -> None:
    calls: list[dict] = []

    class FakeBot:
        async def get_updates(self, **kwargs):
            calls.append(kwargs)
            runner._stopping = True
            return []

    class FakeApplication:
        bot = FakeBot()

    runner = ReliableUpdateRunner(FakeApplication(), tmp_path / "inbox.db")
    await runner._poll_forever()
    runner.store.close()

    assert calls[0]["allowed_updates"] == ALLOWED_UPDATES
    assert {"message", "channel_post", "callback_query"} <= set(ALLOWED_UPDATES)
