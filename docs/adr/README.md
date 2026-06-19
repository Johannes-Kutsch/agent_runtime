# Architecture Decision Records

- [0001 - Runtime ownership migration](0001-runtime-ownership-migration.md): Establish `agent_runtime` as the reusable package boundary for execution contracts, session planning, service selection, and log lifecycle behavior.
- [0002 - Runtime compatibility artifact policy](0002-runtime-compatibility-artifact-policy.md): Keep public names neutral and treat any compatibility-specific artifacts as adapter concerns rather than core runtime concepts.
- [0003 - Stage resolution and failure policy](0003-stage-resolution-and-failure-policy.md): Resolve nested stage chains in priority order and classify provider failures using runtime-owned error categories.
- [0004 - Runtime boundary closure](0004-runtime-boundary-closure.md): Keep application orchestration out of the runtime distribution and require every shipped module to remain standalone-importable.
- [0005 - Runtime public surface narrowing](0005-runtime-public-surface-narrowing.md): Narrow the package root, collapse duplicate runtime entrypoints, and keep work and session seams focused on runtime intent.
- [0006 - Runtime session lifecycle entrypoints](0006-runtime-session-lifecycle-entrypoints.md): Model runtime execution around ephemeral, new-session, and resumed-session lifecycle entrypoints with consumer-owned continuation state.
- [0007 - Session-backed result vocabulary](0007-session-backed-result-vocabulary.md): Use session vocabulary for completed session-backed runtime results and metadata.
- [0008 - Resumed-session request construction](0008-resumed-session-request-construction.md): Keep the canonical resumed-session request name while preserving ordinary continuation and advanced session-plan construction paths.
- [0009 - Built-in provider-only runtime](0009-built-in-provider-only-runtime.md): Ship Claude, Codex, and OpenCode integrations inside the runtime distribution and remove consumer-defined provider services as an extension point.
- [0010 - Portable continuations](0010-portable-continuations.md): Return opaque portable continuation tokens for session-backed runtime execution while leaving durable storage policy to callers.
- [0011 - Live runtime output observation](0011-live-runtime-output-observation.md): Add provider-neutral per-request observation for live agent-message turns without exposing raw provider streams or alternate streaming lifecycle entrypoints.
- [0012 - Live provider smoke runner](0012-live-provider-smoke-runner.md): Add opt-in maintainer tooling that exercises real built-in providers through the Runtime Public Surface without joining the default test suite.
