# Runtime session lifecycle entrypoints

The runtime models execution around session lifecycle rather than the ambiguous one-shot/resumable split: ephemeral execution does not prepare continuity, new-session execution prepares continuation state, and resumed-session execution consumes that continuation without service fallback.

Session-backed execution returns opaque portable continuation tokens. Callers persist and pass them back; they must not depend on provider resume payload schema or runtime-managed state directories. Durable invocation logging follows the same ownership rule: runtime may return structured records, callers own persistence and layout.

## Decision

- Use Ephemeral Run, Start Session Run, and Resume Session Run as canonical lifecycle entrypoints.
- Replace the older one-shot/resumable canonical API split.
- Keep runtime stateless between calls; consumers store or discard continuations.
- Require session-backed success to include meaningful continuation data.
- Use `SessionRunResult` and `SessionRuntimeMetadata` for completed session-backed results and metadata, replacing `ResumableRunResult` and `ResumableRuntimeMetadata`.
- Use `ResumedSessionRunRequest` as the canonical resumed-session request name, with ordinary construction from `Continuation` and advanced construction from lower-level session plans.
- Remove the older `ResumableRunRequest` spelling before release rather than preserving it as a compatibility alias.
- Keep `ToolPolicy` independent from session lifecycle; provider-session continuity keeps one tool policy for its lifetime.
- Let resumed-session execution default model and effort from the continuation while allowing explicit model or effort overrides.
- Do not perform service fallback or reselection during resumed-session execution.
- Limit session-backed execution to built-in providers that can produce and consume portable continuations.
- Put selected service, model, effort, and tool-policy display/policy metadata in result metadata, not the continuation contract.
- Expose provider and service facts such as selected service, account label, reset time, usage, progress, and continuation state; usage-limit grouping stays caller policy outside core runtime API.
- Treat usage limits, cancellation, timeout, temporary service unavailability, and confidently retryable provider failures as normal canonical outcomes with invocation progress, not exceptional failures.

## Consequences

- Public lifecycle language is explicit about whether continuity is created, resumed, or absent.
- Continuations are portable resume tokens, not provider state identifiers relative to runtime directories.
- Providers unable to produce portable continuation data stay limited to ephemeral execution.
- Resumed-session failures do not invalidate a continuation and do not trigger fallback.
- Consumers own application correlation, workflow naming, display grouping, durable logs, and retention.
