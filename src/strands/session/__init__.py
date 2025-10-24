"""Session module.

This module provides session management functionality.
"""

from typing import Any

from .file_session_manager import FileSessionManager
from .repository_session_manager import RepositorySessionManager
from .s3_session_manager import S3SessionManager
from .session_manager import SessionManager
from .session_repository import SessionRepository

__all__ = [
    "DAPR_CONSISTENCY_EVENTUAL",
    "DAPR_CONSISTENCY_STRONG",
    "DaprSessionManager",
    "FileSessionManager",
    "RepositorySessionManager",
    "S3SessionManager",
    "SessionManager",
    "SessionRepository",
]


def __getattr__(name: str) -> Any:
    """Lazy import for optional dependencies."""
    if name == "DaprSessionManager":
        try:
            from .dapr_session_manager import DaprSessionManager

            return DaprSessionManager
        except ModuleNotFoundError as e:
            raise ImportError(
                "DaprSessionManager requires the 'dapr' extra. Install it with: pip install strands-agents[dapr]"
            ) from e

    if name == "DAPR_CONSISTENCY_EVENTUAL":
        try:
            from .dapr_session_manager import DAPR_CONSISTENCY_EVENTUAL

            return DAPR_CONSISTENCY_EVENTUAL
        except ModuleNotFoundError as e:
            raise ImportError(
                "DAPR_CONSISTENCY_EVENTUAL requires the 'dapr' extra. Install it with: pip install strands-agents[dapr]"
            ) from e

    if name == "DAPR_CONSISTENCY_STRONG":
        try:
            from .dapr_session_manager import DAPR_CONSISTENCY_STRONG

            return DAPR_CONSISTENCY_STRONG
        except ModuleNotFoundError as e:
            raise ImportError(
                "DAPR_CONSISTENCY_STRONG requires the 'dapr' extra. Install it with: pip install strands-agents[dapr]"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
