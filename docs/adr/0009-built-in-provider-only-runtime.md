# Built-in provider-only runtime

Status: Partially superseded by [0010 - Portable continuations](0010-portable-continuations.md) for session storage, continuation payloads, durable invocation logging, usage-limit grouping, and caller-defined labels.

`agent_runtime` will ship Claude, Codex, and OpenCode provider integrations inside the runtime distribution and will not support consumer-defined provider services as an extension point.

## Decision

- Consumers select built-in providers through runtime call arguments on a caller-owned `RuntimeClient`.
- `RuntimeClient` owns in-process built-in provider availability and exhaustion state, is safe to reuse concurrently, and does not own durable provider-session storage.
- Provider service objects, service registries, command construction, provider stream parsing, provider-session policy, model/effort allowlists, and provider flag profiles remain runtime-owned internals.
- Runtime constructors on the ordinary consumer surface must not expose execution adapter, service registry, or provider-session adapter injection.
- Built-in execution uses a runtime-owned host subprocess substrate. Application-owned Docker orchestration, dependency installation, managed worktrees, prompt rendering, issue orchestration, and preflight setup stay outside the runtime boundary.
- Built-in provider credentials are supplied per request through immutable `ProviderAuth`, not through process-global setup. Claude uses `ClaudeCodeOAuthToken`, OpenCode uses an API key, and Codex uses host auth files.
- Missing or invalid explicit provider credentials are credential failures and stop execution rather than triggering fallback.
- Session-backed execution is available only for built-in providers that can satisfy the portable continuation contract. New-session and resumed-session calls do not require runtime-managed provider state directories.
- Continuations are opaque portable resume tokens rather than provider state identifiers relative to `RuntimeStateDir`.
- Durable invocation logging is caller-owned. The runtime may return structured invocation records, but it does not own log file layout or retention.
- `RuntimeOutcome` carries top-level optional `ProviderUsage` metadata when a provider reports it. Built-in parsers emit rich usage events rather than treating prompt-token-only events as the usage contract.
- Public provider failure errors may expose provider diagnostic observations, but provider event DTOs remain internal built-in adapter details.
- The migrated built-in invocation behavior should match the existing pycastle Claude, Codex, and OpenCode services so consuming projects can move from pycastle to `agent_runtime` without changing provider invocation semantics.

This supersedes the parts of ADR 0005 that treated `ExecutionProvider`, `ProviderSessionAdapter`, `ServiceRegistry`, and related adapter contracts as public seams for external adapter authors.

## Consequences

- Consuming projects no longer need provider-specific service knowledge to execute prepared agent work.
- The implementation work should migrate pycastle provider invocation code, tests, and required provider dependencies into `agent_runtime`, then slice cleanup around the new `RuntimeClient` consumer surface.
- The current pre-migration adapter protocols may remain temporarily importable while implementation slices land, but they are not the documented consumer extension model.
