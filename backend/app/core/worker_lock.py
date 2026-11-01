"""Advisory worker process locking for single-worker FAISS vector store safety."""
import logging
import os
from pathlib import Path
import sys
from typing import Optional, Union

try:
    import fcntl
except ImportError:
    fcntl = None

logger = logging.getLogger(__name__)


class MultiWorkerError(RuntimeError):
    """Raised when multiple workers attempt to run with a shared local FAISS index."""
    pass


class PlatformNotSupportedError(RuntimeError):
    """Raised when running on an operating system without required platform support."""
    pass


class WorkerLock:
    """Advisory file lock preventing multiple processes from concurrently mutating a local FAISS index.

    FAISS IndexIDMap2 requires in-memory mutations and file-based atomic serialization.
    Running multiple application workers against the same local index directory causes
    vector desynchronization, file clobbering, and silent state divergence.
    This lock enforces single-worker execution at the OS level.
    """

    LOCK_FILENAME = "faiss_worker.lock"

    def __init__(self, lock_dir: Union[str, Path]):
        self.lock_dir = Path(lock_dir).resolve()
        self.lock_file_path = self.lock_dir / self.LOCK_FILENAME
        self._file_obj: Optional[object] = None
        self._acquire_count: int = 0

    @property
    def is_acquired(self) -> bool:
        """Returns True if the lock is held by this WorkerLock instance."""
        return self._acquire_count > 0 and self._file_obj is not None

    def acquire(self) -> bool:
        """Acquires a non-blocking exclusive advisory lock on the lock file.

        Within the same process instance, acquisitions are reference-counted so that
        nested lifespan contexts (e.g. in test suites) do not raise false conflicts.
        Cross-process or concurrent instances will raise MultiWorkerError.

        Returns:
            True if the lock was acquired.

        Raises:
            PlatformNotSupportedError: If run on an unsupported platform without POSIX advisory locking.
            MultiWorkerError: If another process currently holds the lock.
        """
        if fcntl is None or sys.platform == "win32":
            raise PlatformNotSupportedError(
                f"Advisory worker file locking is not supported on platform '{sys.platform}'. "
                f"FAISS single-worker execution currently requires POSIX advisory file locking (fcntl.flock). "
                f"On Windows, run ResearchLens AI within WSL2 or a Linux container."
            )

        if self._acquire_count > 0 and self._file_obj is not None:
            self._acquire_count += 1
            return True

        self.lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            f = open(self.lock_file_path, "a+")
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            f.seek(0)
            f.truncate()
            f.write(f"pid: {os.getpid()}\n")
            f.flush()
            self._file_obj = f
            self._acquire_count = 1
            logger.info(
                "FAISS VectorStore acquired worker lock (PID %d) at '%s'. "
                "Running in required single-worker mode.",
                os.getpid(),
                self.lock_file_path,
            )
            return True
        except (BlockingIOError, OSError) as exc:
            logger.error(
                "FAISS worker lock acquisition failed on '%s': another worker process is active.",
                self.lock_file_path,
            )
            raise MultiWorkerError(
                f"FAISS vector store requires single-worker execution. "
                f"Another process currently holds the exclusive lock on '{self.lock_file_path}'. "
                f"Running multiple Uvicorn/FastAPI workers with a shared local FAISS index causes "
                f"data corruption and vector desynchronization. Launch Uvicorn with --workers 1."
            ) from exc

    def release(self) -> None:
        """Releases the advisory lock and closes the lock file when reference count reaches zero."""
        if self._acquire_count > 1:
            self._acquire_count -= 1
            return

        if self._file_obj is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self._file_obj.fileno(), fcntl.LOCK_UN)
                self._file_obj.close()
            except Exception as e:
                logger.warning("Error releasing worker lock on '%s': %s", self.lock_file_path, e)
            finally:
                self._file_obj = None
                self._acquire_count = 0
                logger.debug("Released worker lock at '%s'", self.lock_file_path)


_global_worker_lock: Optional[WorkerLock] = None


def get_worker_lock(lock_dir: Optional[Union[str, Path]] = None) -> WorkerLock:
    """Returns the process-level WorkerLock singleton."""
    global _global_worker_lock
    if _global_worker_lock is None:
        from app.core.config import settings
        target_dir = lock_dir or settings.DATA_DIR
        _global_worker_lock = WorkerLock(target_dir)
    return _global_worker_lock


def reset_worker_lock() -> None:
    """Resets the global worker lock singleton (primarily for tests)."""
    global _global_worker_lock
    if _global_worker_lock is not None:
        _global_worker_lock.release()
        _global_worker_lock = None
