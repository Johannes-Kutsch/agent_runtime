# TransientAgentError retired: non-retryable transient signals collapse into HardAgentError

Status: active

`ar` raised `TransientAgentError` for `TransientError` provider events whose `classification` was anything other than `"retryable"`. The retryable path already surfaced as `ProviderUnavailableError` (an outcome-dispatch value per ADR 0012). This left `TransientAgentError` in an awkward middle ground: the type name says "transient" but the classification says "not retryable" — which is a hard failure in disguise. ADR 0012 drew the line clearly: expected temporary failures are return values; hard failures are exceptions. A non-retryable transient signal belongs on the hard-failure side of that line.

`TransientAgentError` also carried a `status_code` field sourced from the provider event, making it the only ar exception type that propagated an HTTP status code. `HardAgentError` and `AgentCredentialFailureError` already dropped `status_code` at their raise sites; the inconsistency produced dead routing branches in consumers that assumed `status_code` was a reliable field. The correct routing signal across all hard and credential failures is `classification`.

We retire `TransientAgentError` and collapse its raise site into `HardAgentError`. `ar`'s `errors` module contains only exceptions `ar` itself raises (established by ADR 0018).

## Considered Options

- **Document the dual path**: add a note to CONTEXT.md explaining that `TransientAgentError` can be raised directly for non-retryable transient signals. Rejected: documentation institutionalises a design inconsistency rather than fixing it. The type has exactly one raise site; eliminating the asymmetry is cheaper than explaining it forever.
- **Collapse into `HardAgentError`**: raise `HardAgentError(message=event.raw_message, service_name=provider, classification=event.classification)` for all non-retryable `TransientError` events. `TransientAgentError` loses its only raise site and is retired. Accepted.

## Consequences

- `TransientAgentError` is deleted from `errors.py`, `errors.__all__`, and the root `agent_runtime` export. Importing it from `agent_runtime` raises `AttributeError`.
- Consumers that caught `TransientAgentError` must catch `HardAgentError` instead (or its subclass `AgentCredentialFailureError` for credential failures). The `classification` field on `HardAgentError` carries the same routing signal that `TransientAgentError.classification` carried.
- `status_code` disappears from ar's exception surface entirely. The boundary rule added alongside this ADR makes the omission explicit: provider HTTP status codes are provider-internal details; callers route on `classification` strings.
- The retryable `TransientError` path (`classification="retryable"` → `ProviderUnavailableError`) is unchanged.
