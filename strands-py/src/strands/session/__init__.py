"""Session module.

This module provides session management functionality.

``SnapshotSessionManager`` persists a single agent as a versioned snapshot to a unified
:class:`~strands.storage.storage.Storage` backend and supports checkpointing (time-travel
restore). The repository-based managers (``FileSessionManager``, ``S3SessionManager``,
``RepositorySessionManager``) persist each message individually and also cover multi-agent
orchestrators.
"""

from .file_session_manager import FileSessionManager
from .repository_session_manager import RepositorySessionManager
from .s3_session_manager import S3SessionManager
from .session_manager import SessionManager
from .session_repository import SessionRepository
from .snapshot_session_manager import SaveLatestStrategy, SnapshotSessionManager, SnapshotTrigger

__all__ = [
    "FileSessionManager",
    "RepositorySessionManager",
    "S3SessionManager",
    "SaveLatestStrategy",
    "SessionManager",
    "SessionRepository",
    "SnapshotSessionManager",
    "SnapshotTrigger",
]
