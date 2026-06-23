# Stage resolution and failure policy

Status: Stage-chain resolution superseded by [0010 - Single-candidate provider selection](0010-single-candidate-provider-selection.md). Provider failure classification below remains current.

Provider failures are classified by runtime-owned error types so execution distinguishes transient failures, hard failures, credential failures, timeout-like aborts, and usage limits.

## Decision

- Classify provider failures with runtime-owned error types that preserve provider payloads.

## Consequences

- Failure handling stays consistent across provider implementations.
- Runtime makes continuation decisions without knowing application semantics.
