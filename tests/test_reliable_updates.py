"""Tests for the durable Telegram update inbox."""

from pathlib import Path

import httpx
import pytest
from telegram import Update
from telegram.error import TimedOut

from src.bot.reliable import (
    ALLOWED_UPDATES,
    MAX_RETRY_ATTEMPTS,
    ReliableUpdateRunner,
    UpdateStore,
    retry_delay,
)
from src.utils.retry import RetryableJobError, TerminalJobError, is_retryable_error


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


class _RecordingBot:
    """Fake bot that records replies; optionally fails every send."""

    def __init__(self, fail_send: bool = False) -> None:
        self.sent: list[dict] = []
        self.fail_send = fail_send

    async def send_message(self, **kwargs):
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append(kwargs)


def _make_runner(tmp_path: Path, error: Exception, bot: _RecordingBot) -> ReliableUpdateRunner:
    class FakeApplication:
        pass

    application = FakeApplication()
    application.bot = bot
    runner = ReliableUpdateRunner(application, tmp_path / "inbox.db")

    async def dispatch(update: Update) -> None:
        runner._stopping = True
        raise error

    runner._dispatch = dispatch
    return runner


async def _run_one_attempt(runner: ReliableUpdateRunner) -> None:
    runner._stopping = False
    runner.store._connection.execute("UPDATE update_jobs SET next_attempt_at = 0")
    runner.store._connection.commit()
    await runner._worker(0)


def _row(runner: ReliableUpdateRunner):
    return runner.store._connection.execute(
        "SELECT state, attempts FROM update_jobs WHERE update_id = 5"
    ).fetchone()


@pytest.mark.asyncio
async def test_retry_notice_sent_only_on_first_attempt(tmp_path: Path) -> None:
    bot = _RecordingBot()
    runner = _make_runner(tmp_path, RetryableJobError("asr down"), bot)
    try:
        await runner.store.ingest([make_update(5)])
        for _ in range(3):
            await _run_one_attempt(runner)

        assert len(bot.sent) == 1
        assert bot.sent[0]["chat_id"] == 123
        assert "重试" in bot.sent[0]["text"]
        assert _row(runner)["attempts"] == 3
        assert _row(runner)["state"] == "retry"
    finally:
        runner.store.close()


@pytest.mark.asyncio
async def test_retry_notice_found_through_exception_chain(tmp_path: Path) -> None:
    bot = _RecordingBot()
    wrapper = ValueError("wrapped")
    wrapper.__cause__ = RetryableJobError("asr down")
    runner = _make_runner(tmp_path, wrapper, bot)
    try:
        await runner.store.ingest([make_update(5)])
        await _run_one_attempt(runner)
        assert len(bot.sent) == 1
        assert _row(runner)["state"] == "retry"
    finally:
        runner.store.close()


@pytest.mark.asyncio
async def test_other_retryable_error_sends_no_notice_and_failed_send_keeps_retry(
    tmp_path: Path,
) -> None:
    bot = _RecordingBot()
    runner = _make_runner(tmp_path, httpx.ConnectError("boom"), bot)
    try:
        await runner.store.ingest([make_update(5)])
        await _run_one_attempt(runner)
        assert bot.sent == []
        assert _row(runner)["state"] == "retry"
    finally:
        runner.store.close()

    failing = _RecordingBot(fail_send=True)
    runner = _make_runner(tmp_path / "second", RetryableJobError("asr down"), failing)
    try:
        await runner.store.ingest([make_update(5)])
        await _run_one_attempt(runner)
        assert _row(runner)["state"] == "retry"
    finally:
        runner.store.close()


@pytest.mark.asyncio
async def test_retryable_error_at_cap_fails_and_sends_final_reply(tmp_path: Path) -> None:
    assert MAX_RETRY_ATTEMPTS == 100
    bot = _RecordingBot()
    runner = _make_runner(tmp_path, RetryableJobError("asr down"), bot)
    try:
        await runner.store.ingest([make_update(5)])
        runner.store._connection.execute(
            "UPDATE update_jobs SET state = 'retry', attempts = ?", (MAX_RETRY_ATTEMPTS - 1,)
        )
        runner.store._connection.commit()

        await _run_one_attempt(runner)

        assert _row(runner)["state"] == "failed"
        assert _row(runner)["attempts"] == MAX_RETRY_ATTEMPTS
        assert len(bot.sent) == 1
        assert "失败" in bot.sent[0]["text"]
    finally:
        runner.store.close()
