"""Durable Telegram update ingestion and retrying execution."""

import asyncio
import json
import logging
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from telegram import Update
from telegram.ext import Application, ApplicationHandlerStop

from src.utils.retry import RetryableJobError, is_retryable_error, retry_after_seconds

logger = logging.getLogger(__name__)

LEASE_SECONDS = 15 * 60
RETRY_DELAYS_SECONDS = (2, 5, 15, 45, 120, 300)
RETRY_DELAY_CAP_SECONDS = 15 * 60
# attempts is incremented at claim: the 8th run is the last (7 retries, about 23 minutes).
MAX_RETRY_ATTEMPTS = 8
RETRY_NOTICE_TEXT = "⚠️ 处理暂时失败，将自动重试"
GIVE_UP_NOTICE_TEXT = "❌ 处理失败，请稍后重新发送"
# Telegram keeps the allowed_updates of the last getUpdates call made with this token,
# even by another program. Ask for every type on each call so that such a call cannot
# silently narrow what this bot receives.
ALLOWED_UPDATES = list(Update.ALL_TYPES)


@dataclass(frozen=True)
class DurableJob:
    """One persisted inbound Telegram update ready for processing."""

    update_id: int
    payload: dict[str, object]
    attempts: int


class UpdateStore:
    """Small SQLite inbox with an atomic Telegram polling checkpoint."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._lock = asyncio.Lock()
        self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS update_jobs (
                update_id INTEGER PRIMARY KEY,
                payload TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL DEFAULT 0,
                lease_until REAL,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS update_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    async def get_offset(self) -> int | None:
        async with self._lock:
            row = self._connection.execute(
                "SELECT value FROM update_meta WHERE key = 'polling_offset'"
            ).fetchone()
            return int(row["value"]) if row else None

    async def ingest(self, updates: list[Update]) -> None:
        """Persist updates and their next polling offset in one transaction."""
        if not updates:
            return
        now = time.time()
        async with self._lock:
            with self._connection:
                for update in updates:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO update_jobs
                            (update_id, payload, created_at, updated_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (update.update_id, json.dumps(update.to_dict()), now, now),
                    )
                next_offset = max(update.update_id for update in updates) + 1
                self._connection.execute(
                    """
                    INSERT INTO update_meta(key, value) VALUES ('polling_offset', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (str(next_offset),),
                )

    async def claim(self) -> DurableJob | None:
        """Lease a due task. Expired leases are retried after a process restart."""
        now = time.time()
        async with self._lock:
            with self._connection:
                row = self._connection.execute(
                    """
                    SELECT update_id, payload, attempts
                    FROM update_jobs
                    WHERE (state IN ('pending', 'retry') AND next_attempt_at <= ?)
                       OR (state = 'processing' AND lease_until <= ?)
                    ORDER BY update_id
                    LIMIT 1
                    """,
                    (now, now),
                ).fetchone()
                if row is None:
                    return None
                self._connection.execute(
                    """
                    UPDATE update_jobs
                    SET state = 'processing', attempts = attempts + 1,
                        lease_until = ?, updated_at = ?
                    WHERE update_id = ?
                    """,
                    (now + LEASE_SECONDS, now, row["update_id"]),
                )
                return DurableJob(
                    update_id=row["update_id"],
                    payload=json.loads(row["payload"]),
                    attempts=row["attempts"] + 1,
                )

    async def complete(self, update_id: int) -> None:
        await self._set_state(update_id, "complete", None, 0)

    async def fail(self, update_id: int, error: BaseException) -> None:
        await self._set_state(update_id, "failed", str(error), 0)

    async def retry(self, job: DurableJob, error: BaseException) -> float:
        requested = retry_after_seconds(error)
        delay = requested if requested is not None else retry_delay(job.attempts)
        await self._set_state(job.update_id, "retry", str(error), time.time() + delay)
        return delay

    async def _set_state(
        self, update_id: int, state: str, error: str | None, next_attempt_at: float
    ) -> None:
        async with self._lock:
            with self._connection:
                self._connection.execute(
                    """
                    UPDATE update_jobs
                    SET state = ?, last_error = ?, next_attempt_at = ?, lease_until = NULL,
                        updated_at = ?
                    WHERE update_id = ?
                    """,
                    (
                        state,
                        error[:1000] if error else None,
                        next_attempt_at,
                        time.time(),
                        update_id,
                    ),
                )

    def close(self) -> None:
        self._connection.close()


def retry_delay(attempt: int) -> float:
    """Bounded exponential backoff with jitter for retryable jobs."""
    base = RETRY_DELAYS_SECONDS[min(attempt - 1, len(RETRY_DELAYS_SECONDS) - 1)]
    if attempt > len(RETRY_DELAYS_SECONDS):
        base = RETRY_DELAY_CAP_SECONDS
    return base * random.uniform(0.8, 1.2)


class ReliableUpdateRunner:
    """Poll Telegram only after updates can be durably recorded, then process them."""

    def __init__(
        self, application: Application, database_path: Path, worker_count: int = 2
    ) -> None:
        self.application = application
        self.store = UpdateStore(database_path)
        self.worker_count = worker_count
        self._wake_workers = asyncio.Event()
        self._stopping = False

    async def run(self) -> None:
        await self.application.initialize()
        await self.application.start()
        workers = [asyncio.create_task(self._worker(index)) for index in range(self.worker_count)]
        try:
            await self._poll_forever()
        finally:
            self._stopping = True
            self._wake_workers.set()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await self.application.stop()
            await self.application.shutdown()
            self.store.close()

    async def _poll_forever(self) -> None:
        while not self._stopping:
            try:
                updates = await self.application.bot.get_updates(
                    offset=await self.store.get_offset(),
                    timeout=30,
                    allowed_updates=ALLOWED_UPDATES,
                )
                await self.store.ingest(updates)
                if updates:
                    logger.info("Durably ingested %s Telegram update(s)", len(updates))
                    self._wake_workers.set()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if is_retryable_error(error):
                    logger.warning("Telegram polling temporarily failed; retrying: %s", error)
                    await asyncio.sleep(retry_delay(1))
                    continue
                logger.exception("Telegram polling failed unexpectedly; retrying conservatively")
                await asyncio.sleep(RETRY_DELAY_CAP_SECONDS)

    async def _worker(self, worker_index: int) -> None:
        while not self._stopping:
            job = await self.store.claim()
            if job is None:
                self._wake_workers.clear()
                try:
                    await asyncio.wait_for(self._wake_workers.wait(), timeout=2)
                except TimeoutError:
                    pass
                continue
            update: Update | None = None
            try:
                update = Update.de_json(job.payload, self.application.bot)
                await self._dispatch(update)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if is_retryable_error(error) and job.attempts >= MAX_RETRY_ATTEMPTS:
                    await self.store.fail(job.update_id, error)
                    logger.error(
                        "Update_id=%s given up after %s attempts: %s",
                        job.update_id,
                        job.attempts,
                        error,
                    )
                    await self._notify(update, GIVE_UP_NOTICE_TEXT)
                elif is_retryable_error(error):
                    delay = await self.store.retry(job, error)
                    logger.warning(
                        "Retrying update_id=%s after %.1fs (attempt=%s): %s",
                        job.update_id,
                        delay,
                        job.attempts,
                        error,
                    )
                    if job.attempts == 1 and _has_retryable_job_error(error):
                        await self._notify(update, RETRY_NOTICE_TEXT)
                else:
                    await self.store.fail(job.update_id, error)
                    logger.exception("Update_id=%s failed permanently", job.update_id)
            else:
                await self.store.complete(job.update_id)
                logger.info("Consumed update_id=%s with worker=%s", job.update_id, worker_index)

    async def _notify(self, update: Update | None, text: str) -> None:
        """Best-effort reply to the chat of the update; failures never affect the job."""
        chat = update.effective_chat if update is not None else None
        if chat is None:
            return
        try:
            await self.application.bot.send_message(chat_id=chat.id, text=text)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("Could not send job notice: %s", type(error).__name__)

    async def _dispatch(self, update: Update) -> None:
        """Run the normal PTB route selection but let errors reach the durable worker."""
        context = None
        for handlers in [handlers.copy() for handlers in self.application.handlers.values()]:
            for handler in handlers:
                check_result = handler.check_update(update)
                if check_result is None or check_result is False:
                    continue
                if context is None:
                    context = self.application.context_types.context.from_update(
                        update, self.application
                    )
                    await context.refresh_data()
                try:
                    await handler.handle_update(update, self.application, check_result, context)
                except ApplicationHandlerStop:
                    return
                break


def _has_retryable_job_error(error: BaseException) -> bool:
    """Whether a RetryableJobError is anywhere in the __cause__/__context__ chain."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, RetryableJobError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False
