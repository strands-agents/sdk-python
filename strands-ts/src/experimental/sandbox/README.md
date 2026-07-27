# Modal Sandbox (Experimental)

`ModalSandbox` adapts a caller-owned Modal Sandbox to the Strands `Sandbox`
interface. The API remains experimental while its lifecycle and cancellation
behavior are validated in production workloads.

Per-command cleanup targets the command's process group and descendants tagged
by the adapter. It cannot contain code that deliberately removes the tag and
detaches itself. Hostile workloads require a dedicated Modal Sandbox that the
caller terminates as a whole.

Images must provide a POSIX shell, Linux `/proc`, and `util-linux` `setsid`
with `--fork` and `--wait`.

## Graduation Criteria

- Run live integration tests against every supported Modal SDK minor release.
- Validate command execution, cancellation, and binary file workflows in at
  least three production use cases.
- Confirm that the public constructor and lifecycle ownership model need no
  breaking changes.

Remove the integration if Modal cannot provide reliable per-command
cancellation without terminating the caller-owned Sandbox.
