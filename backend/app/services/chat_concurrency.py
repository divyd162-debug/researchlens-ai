"""In-memory concurrency coordination for chat sessions in single-worker mode."""
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import threading
import time
from typing import Dict, Generator, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


@dataclass
class _ReservationEntry:
    """Internal tracking record for an active session reservation."""
    operation: str
    started_at: float
    thread_ident: int
    token: UUID
    is_active: bool = True
    in_context: bool = False


class ChatSessionBusyError(Exception):
    """Raised when an operation conflicts with an active generation or deletion on a session."""

    def __init__(self, message: str, code: str = "SESSION_BUSY"):
        super().__init__(message)
        self.code = code


class ChatConcurrencyManager:
    """Coordinates concurrent operations per chat session within a worker process.

    ResearchLens AI enforces single-worker execution via WorkerLock (due to local FAISS index
    synchronization requirements). Within the single worker, requests are executed concurrently
    across worker threads in an AnyIO thread pool.

    This manager guarantees:
    1. A session cannot have overlapping generations.
    2. A session cannot be deleted while generation or persistence is in progress.
    3. Generation cannot start while a session is actively being deleted.
    4. Two independent sessions never block one another.
    5. Clean release in all failure modes (exceptions, timeouts, cancellations).
    """

    def __init__(self, stale_timeout_seconds: float = 300.0):
        self._lock = threading.Lock()
        # Maps session_id -> _ReservationEntry
        self._active_sessions: Dict[UUID, _ReservationEntry] = {}
        self._stale_timeout_seconds = stale_timeout_seconds

    def is_busy(self, session_id: UUID) -> bool:
        """Checks if a session is currently reserved for any operation."""
        with self._lock:
            self._cleanup_stale_locked()
            return session_id in self._active_sessions

    def get_active_operation(self, session_id: UUID) -> Optional[str]:
        """Returns the active operation name ('GENERATING' or 'DELETING') if busy, else None."""
        with self._lock:
            self._cleanup_stale_locked()
            if session_id in self._active_sessions:
                return self._active_sessions[session_id].operation
            return None

    def reserve(self, session_id: UUID, operation: str) -> _ReservationEntry:
        """Atomically reserves a session for generation or deletion.

        Note:
            Production endpoints should use `session_reservation()` context manager
            to ensure structured lifecycle management. Direct calls must be paired
            with `release()`.

        Args:
            session_id: Unique UUID of the chat session.
            operation: 'GENERATING' or 'DELETING'.

        Returns:
            _ReservationEntry: Active tracking record with unique reservation token.

        Raises:
            ChatSessionBusyError: If the session is already busy with an operation.
        """
        with self._lock:
            self._cleanup_stale_locked()
            if session_id in self._active_sessions:
                entry = self._active_sessions[session_id]
                current_op = entry.operation
                if current_op == "GENERATING":
                    if operation == "DELETING":
                        raise ChatSessionBusyError(
                            "Cannot delete conversation while message generation is in progress.",
                            code="SESSION_BUSY",
                        )
                    else:
                        raise ChatSessionBusyError(
                            "A conversation turn is already in progress for this session.",
                            code="SESSION_BUSY",
                        )
                elif current_op == "DELETING":
                    raise ChatSessionBusyError(
                        "Cannot perform operations while session deletion is in progress.",
                        code="SESSION_BUSY",
                    )
                else:
                    raise ChatSessionBusyError(
                        f"Session is currently busy with operation '{current_op}'.",
                        code="SESSION_BUSY",
                    )

            entry = _ReservationEntry(
                operation=operation,
                started_at=time.monotonic(),
                thread_ident=threading.get_ident(),
                token=uuid4(),
                is_active=True,
                in_context=False,
            )
            self._active_sessions[session_id] = entry
            logger.debug("Reserved session %s for %s [token: %s]", session_id, operation, entry.token)
            return entry

    def release(self, session_id: UUID, entry: Optional[_ReservationEntry] = None) -> None:
        """Releases the reservation on the specified session.

        If a specific reservation entry is passed, only releases if it matches the
        active entry. This prevents superseded or delayed operations from accidentally
        releasing newer reservations.
        """
        with self._lock:
            current_entry = self._active_sessions.get(session_id)
            if current_entry is None:
                if entry is not None:
                    entry.is_active = False
                    entry.in_context = False
                return

            if entry is not None and current_entry is not entry:
                entry.is_active = False
                entry.in_context = False
                logger.warning(
                    "Ignored stale release attempt for session %s from a superseded operation.",
                    session_id,
                )
                return

            popped = self._active_sessions.pop(session_id, None)
            if popped:
                popped.is_active = False
                popped.in_context = False
                logger.debug("Released reservation for session %s (%s)", session_id, popped.operation)

    @contextmanager
    def session_reservation(self, session_id: UUID, operation: str) -> Generator[_ReservationEntry, None, None]:
        """Context manager guaranteeing reservation acquisition and release in all exit paths.

        Marks the reservation entry as actively held inside an active execution boundary
        (`in_context=True`) for the duration of the context.
        """
        entry = self.reserve(session_id, operation)
        entry.in_context = True
        try:
            yield entry
        finally:
            entry.in_context = False
            self.release(session_id, entry=entry)

    def _cleanup_stale_locked(self) -> None:
        """Evicts reservations that have exceeded stale_timeout_seconds without an active operation.

        Security & Safety Invariants:
        1. Correctness over eviction: An active reservation must NEVER be evicted while its
           operation is actively running inside a session_reservation context in a live thread,
           because the operation could still modify conversation state in the database.
        2. Thread identifiers alone do not prove an operation is still running: AnyIO worker
           threads are pooled and remain alive across requests. If a reservation was abandoned
           without an active session_reservation context (`not entry.in_context`), it is safely
           evicted after stale_timeout_seconds even if the worker thread is still alive in the pool.
        3. If the worker thread that acquired the reservation has terminated, the reservation
           is orphaned and cannot modify state; it is evicted after stale_timeout_seconds.
        """
        now = time.monotonic()
        alive_threads = {t.ident for t in threading.enumerate() if t.is_alive()}
        stale_ids = []

        for sid, entry in list(self._active_sessions.items()):
            elapsed = now - entry.started_at
            if elapsed > self._stale_timeout_seconds:
                is_thread_alive = entry.thread_ident in alive_threads
                # Preserve only if demonstrably still actively executing inside context in a live thread
                if entry.is_active and entry.in_context and is_thread_alive:
                    logger.debug(
                        "Reservation for session %s is still actively running in thread %s (duration: %.1fs). Preserving reservation.",
                        sid,
                        entry.thread_ident,
                        elapsed,
                    )
                    continue
                stale_ids.append(sid)

        for sid in stale_ids:
            entry = self._active_sessions.pop(sid, None)
            if entry:
                entry.is_active = False
                entry.in_context = False
                logger.warning(
                    "Evicted stale reservation for session %s (op: %s, duration: %.1fs, thread: %s, in_context: %s)",
                    sid,
                    entry.operation,
                    now - entry.started_at,
                    entry.thread_ident,
                    entry.in_context,
                )


# Process-wide default concurrency coordinator
chat_concurrency_manager = ChatConcurrencyManager()
