# Single-candidate provider selection

Status: Accepted

Runtime provider selection is one explicit `ProviderSelection` per invocation: service, model, effort, and provider credentials. The runtime accepts no nested fallback chains, does not classify fallback eligibility, and performs no provider fallback inside one call. Consuming projects that want fallback start a separate runtime invocation and own the attempt path, because that orchestration is application policy, not runtime selection behavior.

## Consequences

- `ProviderSelection` replaces `StageSelection`, `StageOverride`, and stage-chain vocabulary on the public surface; request fields use `provider_selection`, not `stage` or `override`.
- Runtime results report selected provider facts for one invocation, not fallback attempt paths.
- Resume Session Run derives service, model, effort, and tool policy from the continuation rather than accepting resume-time provider-selection overrides.
