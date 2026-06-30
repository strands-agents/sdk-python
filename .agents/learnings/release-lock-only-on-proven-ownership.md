# Release a lock only on proven ownership, not a mode/config proxy

- **Source PR:** strands-agents/harness-sdk#2954
- **Run:** learning-run-pr2954

When reviewing a fix to a locking/concurrency bug, verify it enforces the real ownership / acquire-release pairing invariant rather than a proxy for it. A lock must be released only by the code path that *proved* it acquired the lock (mirror the acquire result), never based on a `mode`/config flag re-derived elsewhere.

PR #2954 fixed a bug where `Agent.stream_async`'s `finally` could release the invocation lock owned by a concurrent direct tool call. The fix gated the release on `self._concurrency.mode == ConcurrentInvocationMode.THROW` — a proxy that re-derives the acquire decision `begin()` already made. The codebase already had the correct pattern next door: `tools/_caller.py` records the actual acquisition (`acquired_lock = ... try_acquire_lock()`) and releases only `if acquired_lock`. The linked issue #2918 asked for ownership/pairing to be tracked in `_ConcurrencyController`, which the merge did not do.

Also watch for fields whose *name* implies ownership but whose *value* does not track it in every mode: `_BeginResult.lock_acquired` is `True` in `UNSAFE_REENTRANT` mode even though no lock was acquired, so gating release on `begin.lock_acquired` (the intuitive fix) would not have worked. Flag any `release()` conditioned on configuration instead of proven acquisition, and any `lock_acquired`/`owns_lock`-style flag that can be true without a real acquisition.

**Evidence:** PR #2954 (commit e4ad16ea, `agent/agent.py:1217-1218`); root-cause issue #2918; reference pattern in `strands-py/src/strands/tools/_caller.py:97-142`; misleading field at `strands-py/src/strands/agent/_concurrency.py:124-128`; owner-blind primitive at `_concurrency.py:178-181`.