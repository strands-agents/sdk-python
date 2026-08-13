```ts
type MultiAgentSaveLatestStrategy = "node" | "invocation";
```

Defined in: [src/session/session-manager.ts:57](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/session/session-manager.ts#L57)

Controls when `snapshot_latest` is saved for multi-agent orchestrators.

-   `'node'`: after every node invocation completes (default; enables resume from the last completed node after a crash or restart)
-   `'invocation'`: after every orchestrator invocation completes (lower I/O, but only captures state at orchestrator invocation boundaries)