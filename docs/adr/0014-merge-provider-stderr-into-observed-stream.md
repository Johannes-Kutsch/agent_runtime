# Merge provider stderr into the observed output stream

Status: Accepted. Refines ADR 0011 (the "raw output is never dropped" invariant), which only held for stdout.

The internal Built-in Provider Invocation Seam previously read only the child process's stdout and silently discarded stderr. Some providers (notably codex) write usage-limit notices, error text, and "command not found" diagnostics to stderr while leaving stdout empty. With stderr dropped, such a run reduced to empty output and was classified as a clean `Completed` — a real failure reported as success, with nothing in the live feed to show why (this is exactly how a missing codex binary masqueraded as success during a Live Provider Probe run).

## Decision

- The provider output stream the runtime observes and reduces is **stdout and stderr merged**. stderr lines flow through the same path as stdout: the live `Agent Event` feed (via the output-consume hook), the final reduction/classification, and the returned line sequence. This makes ADR 0011's "raw output is never dropped, full raw stream reconstructable by concatenation" actually hold.
- stderr is drained on a side thread (to avoid a pipe-buffer deadlock against stdout) but is handed to the observe/reduce hooks **only from the main thread, batched after stdout completes** — not interleaved in real wall-clock order. Rationale: the `on_live_output` callback is notification-only and not required to be thread-safe (ADR 0011), so emitting it from a single thread preserves that contract. The cost is that, for a provider that genuinely interleaves both streams, stderr lines appear after stdout lines rather than in true order. They are never lost, only reordered.
- Implemented by reading stderr over a pipe and merging in-process, **not** by redirecting the child's stderr into stdout (`stderr=subprocess.STDOUT`). The OS-level redirect is smaller but unobservable through the invocation seam's test double; reading the pipe keeps the merge explicit and verifiable and leaves stderr under runtime control.

## Consequences

- A provider that fails on stderr now surfaces its real text in the live feed and to the classifier, instead of vanishing. Usage-limit / credential / error notices a provider emits on stderr are detected.
- The live feed and any reconstructed raw stream may now contain stderr-shaped text (warnings, tracebacks, locale-specific diagnostics); consumers already treat raw provider output as opaque and own redaction (CONTEXT.md), so no new contract is needed.
- Merging stderr surfaces but does not classify process-level failures: a non-zero exit whose stderr matches no known error pattern still reduces to empty output and is reported as success. That exit-code gap is tracked separately (GitHub issue #247) and is out of scope here.
