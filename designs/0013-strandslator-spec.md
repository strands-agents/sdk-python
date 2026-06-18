# Strandslator Spec

**Status**: Proposed

**Date**: 2026-06-17

**Issue**: https://github.com/strands-agents/harness-sdk/issues/2666

## Overview

This document is a pairing to [0012-strandslator](./0012-strandslator-design.md), which focused on the experimentation behind translation and the context the workflow consumes. Here we specify the system's behaviors and interfaces: what a run takes as input, what it produces, the pipeline that drives the agents, how it runs locally through a CLI, and how that same command runs as a GitHub Action. It is a spec, not an implementation guide. It pins down the externally observable contracts and leaves the internal mechanics to the build.

The guiding principle is that there is one way to run the workflow, and everything else is a wrapper around it. The local CLI is the workflow. The GitHub Action is the local CLI running on a hosted runner. If an automated run fails, a developer has the option to pick up where it left off on their own machine with the same command.

We lean on [`strands-command`](https://github.com/strands-agents/devtools/tree/main/strands-command) as prior art. It already runs Strands agents in GitHub Actions with job separation, permission isolation, artifact-based state handoff, and authorization gating. It is comment-driven (`/strands` on issues and PRs) rather than `workflow_dispatch`-driven, so it is not a drop-in, but we borrow its proven patterns where they fit.

## Contract

A run is defined by what it takes in and what it puts out.

**Input.** A source feature and a target language. Both resolve to locations in the monorepo, and the source carries its paired tests, docstrings, and metadata, plus the strands-md guides, exactly as specified in [0012](./0012-strandslator-design.md). There are no GitHub issues or other out-of-band inputs.

A run also depends on a precondition: the source feature must be **marked ready for translation** per the "Explicit readiness" principle in [0012](./0012-strandslator-design.md). The workflow does not guess whether work is done. How readiness is signalled is left to the feature development workflow; this spec only requires that the signal exists and that a run refuses to start against a source that is not marked ready.

**Output.** A successful run produces a single target-language PR containing:

- The translated code, tests, and docs.
- The review report and its artifacts described in [0012](./0012-strandslator-design.md): behavior traceability matrix, captured test output, differential results, structural map, decision log, dependency and capability delta, sensitive-surface diff, lint/type-check results, and open questions.

The report is the primary interface for the human on the other end. The run's job is not "produce a diff" but "produce a diff a reviewer can approve quickly." Artifacts must be mechanically derived, not narrated.

## Pipeline

The pipeline runs the Plan, Implement, Validate, Document, and Report agents in sequence, handles kickbacks from Validate back to Plan, and surfaces what is happening to a watching human. Per-agent contracts are specified in [0012](./0012-strandslator-design.md).

- **Kickbacks.** When Validate finds a behavior gap or failing test, it sends the run back to Plan with the findings attached. We bound iterations so a run can't loop forever, and surface a clear failure when the bound is hit.
- **Status updates.** Each step emits structured progress (agent, kickback count, pass/fail) so a human can follow along without reading raw transcripts.
- **Human intervention.** A human can pause a run, inspect or amend the in-progress artifacts and context, and resume.
- **Pause and resume across shutdown.** We checkpoint pipeline state after each completed step so a run can be stopped and picked back up from the last checkpoint, whether that's locally or by re-dispatching in CI.

The pipeline reads input from fixed monorepo paths. Agents may gather additional context on their own (e.g. from the web) during a run.

There are two kinds of state to keep straight. **Pipeline checkpoint** is the run's position in the sequence, kickback count, accumulated artifacts, and each agent's output for the next agent to consume. This is what `resume` needs. **Per-agent session state** is one agent's conversation history (`S3SessionManager` in `strands-command`, keyed by `session_id`). It lets a single agent resume mid-conversation but does not capture where the pipeline is. The pipeline checkpoint is persisted at the pipeline level and carried as a GitHub Actions artifact, mirroring how `strands-command` uses artifacts for everything that crosses job boundaries.

## CLI

The workflow runs end to end locally through a single CLI command. This is the primary interface and the thing every other entry point wraps.

- One command takes source feature and target language and runs Plan to Report, producing the translated code, tests, docs, and the review report.
- It runs against the local checkout of the monorepo.
- It supports starting a fresh run, watching status, pausing, and resuming from a checkpoint.

A rough shape:

```
strandslate run --source <feature> --target <language>
strandslate resume <run-id>
strandslate status <run-id>
```

The exact flags will firm up during implementation; the requirement is that resume works after a full shutdown.

## Action

The Action exists because this is where our code lives. It is deliberately thin: the local CLI command running on a hosted runner.

- **Wraps the local command.** The job step is the same `strandslate run` invocation a developer would type.
- **Manual trigger.** `workflow_dispatch`, matching the principle from [0012](./0012-strandslator-design.md) that a human decides when a feature is ready.
- **Source and target inputs.** Exposed as dispatch inputs, passed straight through to the command.
- **Authorization gating.** Following `strands-command`, the workflow gates dispatch against an allowlist of repo roles (`maintain`, `write`, `admin`), with a manual approval gate for anyone else.
- **Permission separation.** The agent runs in a read-only job. Repository changes and deferred GitHub operations are emitted as artifacts, and a separate Finalize job (with write permission, `if: always()`) pushes the branch and opens the PR. This keeps commit and push off the agent's plate and limits blast radius.
- **Resume on failure.** A failed run can be resumed by re-dispatching in CI or by pulling down the checkpoint locally with `gh run download <run-id>` and running `strandslate resume`.

The checkpoint is published via `actions/upload-artifact` in the job's final step (`if: always()`). Artifacts expire (default 90 days, configurable); we set `retention-days` long enough for realistic human handoff. The artifact carries only what `resume` needs; the translated code lands in the PR.

## Context

The pipeline is only as good as the context it reads. Before the workflow can produce reviewable PRs, we need to get the strands-md layer and per-feature metadata into shape.

- **Split up AGENTS.md.** The current files mix concerns. Split into focused guides by role (testing, building, security, documentation) at the fixed paths [0012](./0012-strandslator-design.md) assumes, with shared guidance at the repo root and language-specific variants in each package.
- **Update guidelines with current learnings.** Fold what we've learned from experiments and ongoing port work back into the guides so the agents inherit it.
- **Add code metadata files.** Backfill the per-feature metadata described in [0012](./0012-strandslator-design.md), prioritizing features we intend to translate first.


## References

- **[0012 Strandslator Design](./0012-strandslator-design.md).** The companion design covering the translation workflow, input context, agents, and review artifacts.
- **[Adversarial and cross-language differential testing](https://gist.github.com/agent-of-mkmeral/5a4d0ce16a1242a711d77d7e01c19902#6-adversarial--cross-language-differential-testing).** Design notes on building an agentic multi-language SDK, referenced in [0012](./0012-strandslator-design.md) for the differential testing approach.
- **[strands-command (strands-agents/devtools)](https://github.com/strands-agents/devtools/tree/main/strands-command).** The org's existing action for running Strands agents in GitHub Actions. Prior art for job separation, artifact-based state handoff, and authorization gating.
