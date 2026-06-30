# On a bugfix, distinguish symptom fix from documented root cause

- **Source PR:** strands-agents/harness-sdk#2954
- **Run:** learning-run-pr2954

When a PR fixes a bug that links to an issue documenting a deeper root cause, check whether the change resolves that root cause or only the immediate symptom. If it is a narrower symptom fix, confirm a tracking issue exists for the remaining root cause and flag its absence rather than treating the bug as fully closed by the merge.

Issue #2918 documented the root cause: `_ConcurrencyController.release_lock()` releases the invocation lock whenever it is held, with no check of *which* invocation owns it, and suggested centralizing acquire/release pairing in the controller. PR #2954 instead shipped a minimal fix to the one misbehaving caller (`stream_async`), leaving the owner-blind primitive in place. That is a legitimate, low-risk scoping choice — but a reviewer should confirm the residual root cause is tracked so it is not silently lost when the issue is closed.

**Evidence:** PR #2954 closes #2918, but addresses only the `stream_async` symptom; the root-cause remediation proposed in #2918 (ownership tracking in `_ConcurrencyController`) remains unimplemented in `strands-py/src/strands/agent/_concurrency.py`.