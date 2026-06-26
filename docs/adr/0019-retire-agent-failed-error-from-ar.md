# AgentFailedError belongs to consumers, not ar

Status: active

`ar` defined and exported `AgentFailedError` but never raised it. Every raise site lived in `pycastle`, which already owned a parallel class with the same name and consumer-specific fields (`agent_invocation_log_path`, `session_dir`). A prior grilling session moved the class into `ar` expecting `pycastle` to migrate to it, but that migration required adding a consumer-settable `consumer_log_path` field — a field `ar` would explicitly never populate. That requirement exposed the underlying ownership problem: the class carried consumer-defined semantics and had no business in `ar`.

We retire `AgentFailedError` from `ar`'s public surface. `pycastle` keeps its own class unchanged. `ar`'s `errors` module contains only exceptions `ar` itself raises.

## Considered Options

- **Add `consumer_log_path: str | None` to `AgentFailedError`** (original issue #366): let `ar` own the class and expose a mutable consumer-settable field. Rejected: a field `ar` explicitly never populates is a sign the class does not belong here.
- **Consumer subclassing**: `pycastle` subclasses `ar.AgentFailedError` to add its fields. Rejected: adds a wrapper type with no new behaviour; still requires `ar` to own a class it never raises.
- **Retire**: remove from `ar` entirely; `pycastle` owns its error type. Accepted.

## Consequences

- `pycastle`'s `AgentFailedError` class is unchanged and stays in `pycastle`.
- `ar`'s `errors` module no longer exports `AgentFailedError`; importing it from `agent_runtime` raises `AttributeError`.
- The `InvocationRole` and `SessionNamespace` concepts (fields on the retired class) are removed from `CONTEXT.md`; they were never `Runtime Consumer Surface`.
