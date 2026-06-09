/**
 * Workaround for https://github.com/bytecodealliance/StarlingMonkey/issues/309.
 * If the host's `AbortSignal.any` is broken, replace it. Delete this file
 * once the minimum supported StarlingMonkey version ships the upstream fix.
 */

const probe = AbortSignal.any([new AbortController().signal])
if (probe.aborted) {
  const original = AbortSignal.any
  AbortSignal.any = (signals: Iterable<AbortSignal>): AbortSignal =>
    // @ts-expect-error broken host expects varargs; we adapt by spreading.
    original.call(AbortSignal, ...signals)
}
