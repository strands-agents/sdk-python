```ts
type MultiAgentSaveLatestStrategy = "node" | "invocation";
```

Defined in: [src/session/session-manager.ts:60](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/session/session-manager.ts#L60)

Controls when `snapshot_latest` is saved for multi-agent orchestrators.

-   `'node'`: after every node invocation completes (default; enables resume from the last completed node after a crash or restart)
-   `'invocation'`: after every orchestrator invocation completes (lower I/O, but only captures state at orchestrator invocation boundaries)