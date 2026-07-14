"""Local-filesystem :class:`~strands.storage.Storage` backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from uuid import uuid4

from ..types.exceptions import StorageError
from .base import Storage, namespace, normalize_key, normalize_prefix

if TYPE_CHECKING:
    from ..sandbox.base import Sandbox

_SCRATCH_MARKER = ".__strands_tmp"


class LocalFileStorage:
    """Local-filesystem :class:`~strands.storage.Storage` backend.

    Persists each key as a file under a base directory, mapping the key's ``/``
    segments onto directory segments. On the host filesystem, writes are atomic
    (write to a scratch sibling, then rename) so a crash mid-write never leaves a
    partially written file. When bound to a
    :class:`~strands.sandbox.base.Sandbox` via :meth:`for_sandbox`, all I/O is
    routed through that sandbox instead of the host filesystem (atomicity then
    depends on the sandbox implementation).

    Example:
        ```python
        from strands.storage import LocalFileStorage

        storage = LocalFileStorage("./.strands/")
        await storage.write("sessions/abc/snapshot.json", data)
        ```
    """

    def __init__(self, base_dir: str = "./.strands/", *, sandbox: Sandbox | None = None) -> None:
        """Initialize file-based storage.

        Args:
            base_dir: Root directory under which keys are stored.
            sandbox: Optional sandbox to route I/O through. Usually set via
                :meth:`for_sandbox`.
        """
        self._base_dir = base_dir
        self._sandbox = sandbox

    def for_sandbox(self, sandbox: Sandbox) -> LocalFileStorage:
        """Return a storage instance whose I/O is routed through ``sandbox``.

        Instances already bound to a sandbox return themselves unchanged.

        Args:
            sandbox: Sandbox to route the returned instance's I/O through.

        Returns:
            A ``LocalFileStorage`` with the same base directory, routed through
            ``sandbox``.
        """
        if self._sandbox is not None:
            return self
        return LocalFileStorage(self._base_dir, sandbox=sandbox)

    async def write(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``, overwriting any existing value.

        Args:
            key: Opaque, ``/``-separated key identifying the value.
            data: Raw bytes to persist.

        Raises:
            StorageError: If the key is invalid or the write fails.
        """
        normalized = normalize_key(key)
        path = self._path_for(normalized)
        if self._sandbox is not None:
            try:
                await self._sandbox.write_file(path, bytes(data))
            except Exception as error:
                raise StorageError(f"Failed to write '{normalized}' to sandbox storage") from error
            return
        tmp_path: str | None = None
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = f"{path}{_SCRATCH_MARKER}_{uuid4()}"
            with open(tmp_path, "wb") as file:
                file.write(data)
            os.replace(tmp_path, path)
        except OSError as error:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise StorageError(f"Failed to write '{normalized}' to local storage") from error

    async def read(self, key: str) -> bytes | None:
        """Retrieve the bytes previously stored under ``key``.

        Args:
            key: The key to read.

        Returns:
            The stored bytes, or ``None`` if no value exists for ``key``.

        Raises:
            StorageError: If the key is invalid or the read fails.
        """
        normalized = normalize_key(key)
        path = self._path_for(normalized)
        if self._sandbox is not None:
            try:
                return await self._sandbox.read_file(path)
            except FileNotFoundError:
                return None
            except Exception as error:
                raise StorageError(f"Failed to read '{normalized}' from sandbox storage") from error
        try:
            with open(path, "rb") as file:
                return file.read()
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError as error:
            raise StorageError(f"Failed to read '{normalized}' from local storage") from error

    async def delete(self, key: str) -> None:
        """Delete the value stored under ``key``. A no-op if the key does not exist.

        Args:
            key: The key to delete.

        Raises:
            StorageError: If the key is invalid or the delete fails.
        """
        normalized = normalize_key(key)
        path = self._path_for(normalized)
        if self._sandbox is not None:
            try:
                await self._sandbox.remove_file(path)
            except FileNotFoundError:
                return
            except Exception as error:
                raise StorageError(f"Failed to delete '{normalized}' from sandbox storage") from error
            return
        try:
            os.remove(path)
        except (FileNotFoundError, NotADirectoryError):
            return
        except OSError as error:
            raise StorageError(f"Failed to delete '{normalized}' from local storage") from error

    async def list(self, prefix: str) -> list[str]:
        """List the keys whose names begin with ``prefix``, sorted lexicographically.

        Args:
            prefix: Key prefix to match. An empty string matches all keys.

        Returns:
            The matching keys, sorted ascending.

        Raises:
            StorageError: If the prefix is invalid or the listing fails.
        """
        normalized = normalize_prefix(prefix)
        base = self._base_dir.rstrip("/")
        # Narrow the walk to the deepest directory the prefix fully specifies.
        last_slash = normalized.rfind("/")
        dir_portion = normalized[:last_slash] if last_slash >= 0 else ""
        start_dir = f"{base}/{dir_portion}" if dir_portion else base
        if self._sandbox is not None:
            keys = await _list_keys_sandbox(self._sandbox, start_dir, dir_portion)
        else:
            keys = _list_keys_host(start_dir, dir_portion)
        return sorted(key for key in keys if key.startswith(normalized))

    def _path_for(self, key: str) -> str:
        return f"{self._base_dir.rstrip('/')}/{key}"

    def namespace(self, prefix: str) -> Storage:
        """Return a prefixed view of this storage without mutating the original."""
        return namespace(self, prefix)


def _list_keys_host(dir_path: str, key_prefix: str) -> list[str]:
    """Recursively collect file keys under ``dir_path`` on the host filesystem."""
    try:
        entries = list(os.scandir(dir_path))
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError as error:
        raise StorageError(f"Failed to list local storage under '{key_prefix}'") from error
    found: list[str] = []
    for entry in entries:
        is_dir = entry.is_dir()
        if not is_dir and _SCRATCH_MARKER in entry.name:
            continue
        child_key = f"{key_prefix}/{entry.name}" if key_prefix else entry.name
        if is_dir:
            found.extend(_list_keys_host(f"{dir_path}/{entry.name}", child_key))
        else:
            found.append(child_key)
    return found


async def _list_keys_sandbox(sandbox: Sandbox, dir_path: str, key_prefix: str) -> list[str]:
    """Recursively collect file keys under ``dir_path`` through a sandbox."""
    try:
        entries = await sandbox.list_files(dir_path)
    except FileNotFoundError:
        return []
    except Exception as error:
        raise StorageError(f"Failed to list sandbox storage under '{key_prefix}'") from error
    found: list[str] = []
    for entry in entries:
        if not entry.is_dir and _SCRATCH_MARKER in entry.name:
            continue
        child_key = f"{key_prefix}/{entry.name}" if key_prefix else entry.name
        if entry.is_dir:
            found.extend(await _list_keys_sandbox(sandbox, f"{dir_path}/{entry.name}", child_key))
        else:
            found.append(child_key)
    return found
