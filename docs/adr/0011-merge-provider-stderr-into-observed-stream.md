# Merge provider stderr into the observed output stream

Refines [0009](0009-agent-event-observation-model.md)'s "raw output is never dropped" invariant, which previously only held for stdout. Some providers (notably codex) write usage-limit notices and error text to stderr while leaving stdout empty — with stderr dropped, real failures were classified as `Completed`.

## Decision

- Provider output stream is **stdout and stderr merged**. stderr flows through the same path: live `Agent Event` feed, reduction/classification, returned line sequence.
- stderr drained on a side thread (avoids pipe-buffer deadlock) but handed to observe/reduce hooks **from the main thread, batched after stdout** — not interleaved in wall-clock order. Preserves the single-threaded `on_live_output` contract. stderr lines are never lost, only reordered.
- Implemented by reading stderr over a pipe and merging in-process, **not** `stderr=subprocess.STDOUT`. Keeps the merge explicit, verifiable, and observable through the invocation seam's test double.

## Consequences

- Provider stderr-only failures now surface in the live feed and reach the classifier.
- Live feed may contain stderr-shaped text; consumers already treat raw output as opaque and own redaction.
- Non-zero exit whose stderr matches no known pattern still reduces to empty output (exit-code gap tracked in GitHub issue #247).
