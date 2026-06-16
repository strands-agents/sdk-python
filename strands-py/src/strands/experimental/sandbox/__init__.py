"""Experimental sandbox implementations.

This module ships sandbox backends that depend on optional, separately-installed
packages and whose APIs may change without notice.

- :class:`StrandsShellSandbox` — a :class:`~strands.sandbox.base.Sandbox` backed
  by `Strands Shell <https://github.com/strands-agents/shell>`_, an in-process
  Bourne-compatible shell with declarative filesystem, network, and credential
  mediation. Requires ``pip install strands-agents[shell]``.

The sandbox vends ``sandbox_bash`` and ``sandbox_file_editor`` tools (built from
the :func:`~strands.vended_tools.make_bash` / :func:`~strands.vended_tools.make_file_editor`
factories) via :meth:`StrandsShellSandbox.get_tools`; an agent constructed with
the sandbox registers them automatically.
"""

from .strands_shell import StrandsShellSandbox

__all__ = ["StrandsShellSandbox"]
