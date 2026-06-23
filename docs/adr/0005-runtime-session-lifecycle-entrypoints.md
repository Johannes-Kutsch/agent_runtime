# Runtime session lifecycle entrypoints

Status: Refined by [0010 - Single-candidate provider selection](0010-single-candidate-provider-selection.md) for resume provider identity and request-time credentials.

Runtime models execution around session lifecycle, not the ambiguous one-shot/resumable split: ephemeral execution prepares no continuity, new-session execution prepares continuation state, resumed-session execution consumes that continuation without service fallback.

Session-backed execution returns opaque portable continuation tokens. Callers persist and pass them back; they must not depend on provider resume payload schema or runtime-managed state directories. Durable invocation logging follows the same rule: runtime may return structured records, callers own persistence and layout.

## Decision

- Ephemeral Run, Start Session Run, Resume Session Run are the canonical lifecycle entrypoints, replacing the one-shot/resumable split.
- Runtime stateless between calls; consumers store or discard continuations.
- Session-backed success must include meaningful continuation data.
- `SessionRunResult`/`SessionRuntimeMetadata` for completed session-backed results/metadata (replacing `ResumableRunResult`/`ResumableRuntimeMetadata`).
- `ResumedSessionRunRequest` is the canonical resumed-session request: ordinary construction from `Continuation`, advanced from lower-level session plans. Older `ResumableRunRequest` removed before release, not aliased.
- `ToolPolicy` independent from session lifecycle; continuity keeps one tool policy for its lifetime.
- Resumed-session execution derives service, model, effort, and tool policy from the continuation; credentials received separately. No service fallback or reselection during resume.
- Session-backed execution limited to built-in providers that produce/consume portable continuations.
- Selected service/model/effort and tool-policy display metadata go in result metadata, not the continuation contract.
- Expose provider/service facts (selected service, account label, reset time, usage, progress, continuation state); usage-limit grouping stays caller policy.
- Usage limits, cancellation, timeout, temporary unavailability, and confidently retryable provider failures are normal canonical outcomes with invocation progress, not exceptional failures.

## Consequences

- Lifecycle language is explicit about whether continuity is created, resumed, or absent.
- Continuations are portable resume tokens, not provider state identifiers relative to runtime directories.
- Providers without portable continuation data stay ephemeral-only.
- Resumed-session failures don't invalidate a continuation and don't trigger fallback.
- Consumers own application correlation, workflow naming, display grouping, durable logs, retention.
