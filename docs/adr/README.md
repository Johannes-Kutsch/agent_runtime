# Architecture Decision Records

- [0001 - Runtime ownership migration](0001-runtime-ownership-migration.md): Establish `agent_runtime` as reusable package boundary for execution contracts, session planning, service selection, and lifecycle behavior.
- [0002 - Runtime compatibility artifact policy](0002-runtime-compatibility-artifact-policy.md): Keep public names neutral and treat compatibility-specific artifacts as adapter concerns.
- [0003 - Stage resolution and failure policy](0003-stage-resolution-and-failure-policy.md): Stage-chain resolution superseded by ADR 0010; provider failure classification remains current.
- [0004 - Runtime public surface narrowing](0004-runtime-public-surface-narrowing.md): Narrow package root, collapse duplicate runtime entrypoints, and keep work/session/provider seams focused on runtime intent.
- [0005 - Runtime session lifecycle entrypoints](0005-runtime-session-lifecycle-entrypoints.md): Model execution around ephemeral, new-session, and resumed-session entrypoints with portable consumer-owned continuations; refined by ADR 0010 for resume provider identity.
- [0006 - Built-in provider-only runtime](0006-built-in-provider-only-runtime.md): Ship Claude, Codex, and OpenCode integrations inside runtime distribution and remove consumer-defined provider services as extension point; refined by ADR 0010 for selection, auth, and availability policy.
- [0007 - Live provider smoke runner](0007-live-provider-smoke-runner.md): Add opt-in maintainer tooling that exercises real built-in providers through Runtime Public Surface without joining default test suite.
- [0008 - Provider invocation argv and stdin boundary](0008-provider-invocation-argv-stdin-boundary.md): Model built-in provider invocation as executable arguments plus explicit prompt input rather than host shell command strings.
- [0009 - No documentation regression tests](0009-no-documentation-regression-tests.md): Keep default tests focused on executable behavior instead of documentation or help prose wording.
- [0010 - Single-candidate provider selection](0010-single-candidate-provider-selection.md): Replace stage chains with one provider selection per invocation and leave fallback orchestration to consuming projects.
- [0011 - Typed Agent Events with raw output](0011-typed-agent-events-with-raw-output.md): Replace message-only live turns with typed `Agent Event`s (message / tool call / other) carrying both a filtered view and raw provider output, shared by Live Runtime Output and the finished-run log.
