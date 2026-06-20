# Architecture Decision Records

- [0001 - Runtime ownership migration](0001-runtime-ownership-migration.md): Establish `agent_runtime` as reusable package boundary for execution contracts, session planning, service selection, and lifecycle behavior.
- [0002 - Runtime compatibility artifact policy](0002-runtime-compatibility-artifact-policy.md): Keep public names neutral and treat compatibility-specific artifacts as adapter concerns.
- [0003 - Stage resolution and failure policy](0003-stage-resolution-and-failure-policy.md): Resolve nested stage chains in priority order and classify provider failures with runtime-owned error categories.
- [0004 - Runtime public surface narrowing](0004-runtime-public-surface-narrowing.md): Narrow package root, collapse duplicate runtime entrypoints, and keep work/session/provider seams focused on runtime intent.
- [0005 - Runtime session lifecycle entrypoints](0005-runtime-session-lifecycle-entrypoints.md): Model execution around ephemeral, new-session, and resumed-session entrypoints with portable consumer-owned continuations.
- [0006 - Built-in provider-only runtime](0006-built-in-provider-only-runtime.md): Ship Claude, Codex, and OpenCode integrations inside runtime distribution and remove consumer-defined provider services as extension point.
- [0007 - Live runtime output observation](0007-live-runtime-output-observation.md): Add provider-neutral per-request observation for live agent-message turns without exposing raw provider streams or alternate streaming lifecycle entrypoints.
- [0008 - Live provider smoke runner](0008-live-provider-smoke-runner.md): Add opt-in maintainer tooling that exercises real built-in providers through Runtime Public Surface without joining default test suite.
- [0009 - Provider invocation argv and stdin boundary](0009-provider-invocation-argv-stdin-boundary.md): Model built-in provider invocation as executable arguments plus explicit prompt input rather than host shell command strings.
- [0010 - No documentation regression tests](0010-no-documentation-regression-tests.md): Keep default tests focused on executable behavior instead of documentation or help prose wording.
