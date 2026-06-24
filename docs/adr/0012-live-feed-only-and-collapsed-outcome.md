# Live feed is the only logging interface; collapse the outcome surface

Status: Accepted. Supersedes the finished-run-log parts of ADR 0011 and the invocation-record artifact parts of ADR 0007.

ADR 0011 gave the runtime two observation channels sharing one `Agent Event` vocabulary: an incremental live feed and a finished-run log (`InvocationRecord`) holding the complete ordered event sequence plus terminal metadata and raw provider-output evidence. That at-rest log earns its keep only for consumers that want "give me the whole run" — but the live feed already carries the full raw stream (every chunk maps to an event, "other" included), so the finished-run log is a second copy the runtime must build and the consumer must understand. The outcome value had grown into a flat ~12-field bag where most fields are `None` for any given kind, with property accessors that raise `AttributeError` when read off the wrong kind. We narrow both.

## Decision

- **Live feed is the only output-observation channel.** `on_live_output` emitting `Agent Event`s. No stored, compiled, or finished-run log; no end-of-session output file. Remove `InvocationRecord` and `RuntimeOutcome.invocation_records` from the public surface.
- **`Agent Event` collapses to three fields**: `type` (closed set — agent message / agent tool call / other), a single human-readable `display_message`, and `raw_provider_output`. Drop the fragmented `text`/`tool_name`/`payload`/`descriptor` fields and per-event `service_name`. Structured detail (e.g. tool identity) is available only by reading raw.
- **`RuntimeOutcome` becomes `{ kind, result }`.** `kind` is a discriminated union: `Completed`, `UsageLimited(reset_time)`, `NoServiceAvailable(reset_time)`, `Cancelled`, `TimedOut`, `RetryableProviderFailure`. Bad credentials stay an exception (`AgentCredentialFailureError`), not a kind. Common run facts move to `result`; only genuinely kind-specific data sits on the variant.
- **`RunResult` always present**, even after interruption: `output`, `usage`, `continuation` (none for ephemeral), and `selected`. Unify the ephemeral/session result split into one `RunResult`. Drop `tool_access` (consumer already supplied the policy).
- **`ResolvedProvider`** `{ service, model, effort }` — credential-free, distinct from request-side `ProviderSelection`. Canonical wherever the triple appears: result, continuation payload, session metadata.
- **Remove `account_label`** (never populated — dead caller-label vestige). **Remove `invocation_progress` from the public surface**; keep it internal as the gate that decides whether a continuation is built. Callers infer resumability from continuation presence.

## Consequences

- One observation model, learned once. No second at-rest vocabulary.
- Consumers wanting the whole run buffer the live feed themselves; consumers wanting tool *structure* parse `raw_provider_output`. Both were derivable before; now they are the only path.
- The outcome is a sum type: no `None`-by-default fields, no `AttributeError` accessors, "reset_time on a completed run" unrepresentable. Consumers `match` on `kind` only for terminus detail; common facts always read off `result`.
- Ephemeral Consumer Fallback loses the crisp `NOT_STARTED`/`STARTED` enum; inferred from `output`/`usage` instead.
- Breaking public-surface change. Consumers reading `invocation_records`, `account_label`, the old `Agent Event` fields, or the flat outcome must migrate (smoke runner included). `public-api.md` and the superseded parts of 0011/0007 to be reconciled.
