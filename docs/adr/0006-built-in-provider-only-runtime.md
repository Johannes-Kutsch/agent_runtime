# Built-in provider-only runtime

Status: Refined by [0005 - Runtime session lifecycle entrypoints](0005-runtime-session-lifecycle-entrypoints.md) for continuation payloads, durable logging, usage-limit grouping, and caller-owned application labels.

`agent_runtime` ships Claude, Codex, and OpenCode provider integrations inside the runtime distribution. It does not support consumer-defined provider services as an extension point.

## Decision

- Consumers select built-in providers through runtime call arguments on caller-owned `RuntimeClient`.
- `RuntimeClient` owns in-process built-in provider availability and exhaustion state, is safe to reuse concurrently, and does not own durable provider-session storage.
- Provider service objects, service registries, command construction, provider stream parsing, provider-session policy, model/effort allowlists, and provider flag profiles remain runtime-owned internals.
- Runtime constructors on the ordinary consumer surface must not expose execution adapter, service registry, or provider-session adapter injection.
- Built-in execution uses a runtime-owned host subprocess substrate. Application Docker orchestration, dependency installation, execution-directory management, prompt rendering, issue orchestration, and preflight setup stay outside the runtime boundary.
- Built-in provider credentials are supplied per request through immutable `ProviderAuth`, not process-global setup.
- Claude uses `ClaudeCodeOAuthToken`, OpenCode uses an API key, and Codex uses host auth files.
- Missing or invalid explicit provider credentials are credential failures and stop execution rather than triggering fallback.
- Session-backed execution is available only for built-in providers that satisfy the portable continuation contract.
- Durable invocation logging is caller-owned; runtime may return structured invocation records but does not own log file layout or retention.
- `RuntimeOutcome` carries top-level optional `ProviderUsage` when a provider reports it.
- Built-in parsers emit rich usage events rather than treating prompt-token-only events as the usage contract.
- Public provider failure errors may expose provider diagnostic observations, but provider event DTOs remain internal built-in adapter details.
- Migrated built-in invocation behavior should match existing pycastle Claude, Codex, and OpenCode services.
- Treat `ExecutionProvider`, `ProviderSessionAdapter`, `ServiceRegistry`, and related adapter contracts as internal or advanced runtime seams, not consumer extension points.

## Consequences

- Consuming projects no longer need provider-specific service knowledge to execute prepared agent work.
- Pycastle migration can reuse provider invocation semantics without exposing provider services.
- Pre-migration adapter protocols may remain temporarily importable, but they are not the documented consumer extension model.
