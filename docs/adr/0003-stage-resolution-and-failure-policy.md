# Stage resolution and failure policy

Runtime service selection uses ordered stage chains. Each chain node may name a preferred service and a fallback node. Resolution chooses the first configured and available service, preserving the remaining fallback chain for later evaluation.

Provider failures are classified by runtime-owned error types so execution can distinguish transient failures, hard failures, credential failures, timeout-like aborts, and usage limits.

## Decision

- Resolve nested stage chains in priority order.
- Skip unavailable or unconfigured services instead of treating every chain as a hard failure.
- Classify provider failures with runtime-owned error types that preserve provider payloads.

## Consequences

- Consuming projects can defer provider choice until runtime.
- Failure handling remains consistent across provider implementations.
- The runtime can make continuation decisions without knowing application semantics.
