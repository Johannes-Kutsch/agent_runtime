# Built-in provider-only runtime

Status: Refined by [0005 - Runtime session lifecycle entrypoints](0005-runtime-session-lifecycle-entrypoints.md) for continuation payloads, durable logging, usage-limit grouping, and caller-owned application labels. Refined by [0010 - Single-candidate provider selection](0010-single-candidate-provider-selection.md) for single-candidate selection, credential ownership, and consumer-owned fallback.

`agent_runtime` ships Claude, Codex, and OpenCode provider integrations inside the runtime distribution. It does not support consumer-defined provider services as an extension point.

## Decision

- Consumers select built-in providers through runtime call arguments on caller-owned `RuntimeClient`.
- `RuntimeClient` is safe to reuse concurrently; owns no cross-call provider availability policy or durable provider-session storage.
- Provider service objects, service registries, command construction, stream parsing, provider-session policy, model/effort allowlists, and provider flag profiles stay runtime-owned internals.
- Ordinary consumer constructors must not expose execution adapter, service registry, or provider-session adapter injection.
- Built-in execution uses a runtime-owned host subprocess substrate. Application Docker orchestration, dependency installation, execution-directory management, prompt rendering, issue orchestration, and preflight setup stay outside runtime.
- Built-in provider credentials supplied through `ProviderSelection`, or separately to Resume Session Run (continuations don't store credentials). Claude uses `ClaudeCodeOAuthToken`, OpenCode an API key, Codex host auth files.
- Missing/invalid explicit credentials are credential failures and don't trigger runtime fallback.
- Session-backed execution only for built-in providers satisfying the portable continuation contract.
- Durable invocation logging is caller-owned; runtime may return structured records but owns no log layout or retention.
- `RuntimeOutcome` carries top-level optional `ProviderUsage` when reported; built-in parsers emit rich usage events, not prompt-token-only.
- Public provider failure errors may expose provider diagnostic observations; provider event DTOs stay internal.
- `ExecutionProvider`, `ProviderSessionAdapter`, `ServiceRegistry`, and related adapter contracts are internal/advanced seams, not consumer extension points.

## Consequences

- Consuming projects don't need provider-specific service knowledge to execute prepared agent work.
- They reuse provider invocation semantics through Runtime Public Surface without provider services.
- Pre-migration adapter protocols may remain temporarily importable but are not the documented consumer extension model.
