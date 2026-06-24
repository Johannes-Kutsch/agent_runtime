# Runtime session lifecycle entrypoints

Status: Refined by [0008 - Single-candidate provider selection](0008-single-candidate-provider-selection.md) for resume provider identity and request-time credentials.

Execution modeled around session lifecycle: ephemeral (no continuity), new-session (prepares continuation), resumed-session (consumes continuation without fallback). Session-backed execution returns opaque portable continuation tokens; callers persist and pass them back.

## Decision

- Ephemeral Run, Start Session Run, Resume Session Run are canonical lifecycle entrypoints.
- Runtime stateless between calls; consumers store or discard continuations.
- Session-backed success must include meaningful continuation data.
- `SessionRunResult`/`SessionRuntimeMetadata` for session-backed results/metadata.
- `ResumedSessionRunRequest` is canonical resumed-session request.
- `ToolPolicy` independent from session lifecycle; continuity keeps one tool policy for its lifetime.
- Resume derives service, model, effort, tool policy from continuation; credentials received separately. No fallback or reselection.
- Session-backed execution limited to built-in providers with portable continuations.
- Selected service/model/effort and tool-policy display metadata in result metadata, not continuation contract.
- Usage limits, cancellation, timeout, temporary unavailability are normal outcomes, not exceptional failures.

## Consequences

- Lifecycle language explicit about whether continuity is created, resumed, or absent.
- Continuations are portable resume tokens, not provider state identifiers.
- Providers without portable continuation data stay ephemeral-only.
- Resumed-session failures don't invalidate continuations or trigger fallback.
- Consumers own correlation, workflow naming, display grouping, durable logs, retention.
