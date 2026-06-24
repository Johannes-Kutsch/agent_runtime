# Built-in provider-only runtime

Status: Refined by [0004 - Runtime session lifecycle entrypoints](0004-runtime-session-lifecycle-entrypoints.md) for continuations and logging. Refined by [0008 - Single-candidate provider selection](0008-single-candidate-provider-selection.md) for selection, credentials, and consumer-owned fallback.

`agent_runtime` ships Claude, Codex, and OpenCode integrations. No consumer-defined provider services.

## Decision

- Consumers select built-in providers through runtime call arguments on `RuntimeClient`.
- `RuntimeClient` reusable concurrently; no cross-call provider availability policy or durable storage.
- Provider services, registries, command construction, stream parsing, model/effort allowlists, flag profiles stay internal.
- Consumer constructors must not expose execution adapter, service registry, or provider-session adapter injection.
- Built-in execution uses runtime-owned host subprocess. Docker, dependency install, prompt rendering, issue orchestration stay outside.
- Credentials via `ProviderSelection`, or separately for Resume Session Run. Claude: `ClaudeCodeOAuthToken`, OpenCode: API key, Codex: host auth files.
- Missing/invalid credentials are credential failures, not runtime fallback triggers.
- Session-backed execution only for providers with portable continuations.
- Durable logging caller-owned; runtime owns no log layout or retention.
- `RuntimeOutcome` carries optional `ProviderUsage`; built-in parsers emit rich usage, not prompt-token-only.
- `ExecutionProvider`, `ProviderSessionAdapter`, `ServiceRegistry` are internal seams, not consumer extension points.

## Consequences

- Consumers don't need provider-specific service knowledge.
- Invocation semantics reused through Runtime Public Surface.
- Pre-migration adapter protocols not the documented consumer model.
