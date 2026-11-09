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
    assert MAX_RETRY_ATTEMPTS == 8
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


# ---- health endpoint ----


def _status_and_body(response: bytes) -> tuple[int, str]:
    head, _, body = response.partition(b"\r\n\r\n")
    return int(head.split(b" ")[1]), body.decode()


def test_health_response_by_last_success_age() -> None:
    from src.bot.health import build_response

    now = 1000.0
    get = b"GET /health HTTP/1.1"
    assert _status_and_body(build_response(get, now - 10, 0.0, now)) == (200, "ok")
    assert _status_and_body(build_response(get, now - 120, 0.0, now)) == (503, "stale")


def test_health_response_before_first_success_uses_start_grace() -> None:
    from src.bot.health import build_response

    now = 1000.0
    get = b"GET /health HTTP/1.1"
    assert _status_and_body(build_response(get, None, now - 10, now))[0] == 200
    assert _status_and_body(build_response(get, None, now - 120, now))[0] == 503


def test_health_unknown_path_is_404() -> None:
    from src.bot.health import build_response

    status, _ = _status_and_body(build_response(b"GET /other HTTP/1.1", 1.0, 0.0, 2.0))
    assert status == 404


class _FakeBot:
    def __init__(self, runner_ref: list, outcomes: list) -> None:
        self.runner_ref = runner_ref
        self.outcomes = outcomes

    async def get_updates(self, **kwargs):
        outcome = self.outcomes.pop(0)
        if not self.outcomes:
            self.runner_ref[0]._stopping = True
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeApp:
    def __init__(self, bot) -> None:
        self.bot = bot
        self.running_probe = None

    async def initialize(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


async def _http_get(port: int) -> tuple[int, str]:
    import asyncio

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
    await writer.drain()
    data = await reader.read()
    writer.close()
    return _status_and_body(data)


@pytest.mark.asyncio
async def test_runner_serves_health_while_running_and_closes_after(tmp_path: Path) -> None:
    import asyncio

    ref: list = []
    seen: dict = {}

    class Bot(_FakeBot):
        async def get_updates(self, **kwargs):
            server = runner._health_server
            if server is not None:
                seen["port"] = server.sockets[0].getsockname()[1]
                seen["result"] = await _http_get(seen["port"])
            return await super().get_updates(**kwargs)

    runner = ReliableUpdateRunner(_FakeApp(Bot(ref, [[]])), tmp_path / "inbox.db", health_port=0)
    ref.append(runner)
    await runner.run()

    assert seen.get("result") == (200, "ok")
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", seen["port"])


@pytest.mark.asyncio
async def test_runner_without_health_port_starts_no_server(tmp_path: Path) -> None:
    ref: list = []
    runner = ReliableUpdateRunner(_FakeApp(_FakeBot(ref, [[]])), tmp_path / "inbox.db")
    ref.append(runner)
    seen = {}
    orig = runner._poll_forever

    async def probe():
        seen["server"] = runner._health_server
        await orig()

    runner._poll_forever = probe
    await runner.run()
    assert seen["server"] is None


@pytest.mark.asyncio
async def test_last_success_recorded_only_after_successful_get_updates(tmp_path: Path) -> None:
    ref: list = []
    runner = ReliableUpdateRunner(_FakeApp(_FakeBot(ref, [TimedOut()])), tmp_path / "inbox.db")
    ref.append(runner)

    async def no_sleep(_: float) -> None:
        return None

    import src.bot.reliable as reliable

    original = reliable.asyncio.sleep
    reliable.asyncio.sleep = no_sleep
    try:
        await runner._poll_forever()
    finally:
        reliable.asyncio.sleep = original
    assert runner.last_poll_success is None

    runner._stopping = False
    runner.application.bot.outcomes = [[]]
    await runner._poll_forever()
    runner.store.close()
    assert runner.last_poll_success is not None
