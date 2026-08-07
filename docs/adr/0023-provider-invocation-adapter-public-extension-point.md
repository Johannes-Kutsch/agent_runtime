# Provider Invocation Adapter public extension point

Status: active

pycastle's `_DockerlessRuntimeClient` routes built-in provider invocations through Docker instead of spawning host subprocesses. To do this it needs to inject a custom `ProviderInvocationAdapter` into the full session lifecycle — continuation resolution, provider-state allocation, redirect handling, and policy validation — not just into a single invocation dispatch. Every internal call site already accepts a `provider_invocation_adapter` parameter; the gap was that the public `RuntimeClient` entry-points offered no injection hook, and the four associated types had no stable import path.

## Considered options

### Injection attachment point

Three candidates: `RuntimeClient` constructor, per-request field (alongside `argv_transform`), or per-method parameter.

**Per-request field.** `argv_transform` lives on the request objects because it is a per-call business transform — it changes *what* command runs and may legitimately vary per call site. `ProviderInvocationAdapter` is infrastructure: it changes *how* every call on a client is executed and is stable for the lifetime of the client. Putting infrastructure on a request object conflates two abstraction levels and would require repeating the adapter on every call.

**Per-method parameter.** Adds a parameter to three already-rich method signatures and is inconsistent with `argv_transform`, which is on the request.

**`RuntimeClient` constructor.** Matches how transport layers work in most SDKs. One injection point for all three run kinds; callers that do not need a custom adapter pass nothing. Accepted.

### Export scope

**New `agent_runtime.invocation` module.** Creates a dedicated module with cleaner conceptual separation, but adds a new stable boundary to maintain forever for a small, cohesive set of symbols.

**Flat export into `agent_runtime` root.** Keeps one stable import path. The symbol set is small and cohesive. Accepted.

### `ProviderInvocationFailure` construction

**Factory classmethods** (`ProviderInvocationFailure.usage_limited(...)`, `.provider_unavailable(...)`). Hides `InvocationFailureKind` from the construction path but grows the API surface.

**Direct dataclass constructor + exported `InvocationFailureKind`.** `InvocationFailureKind` is a two-variant enum and stable. Callers construct failures directly, matching how `ProviderInvocationResult` is already constructed. Accepted.

### Streaming hook

**Formal Protocol.** Would require `reduce_output` callables to formally declare `consume_stdout_lines`; the existing production adapter does not implement a formal Protocol.

**Public alias of the existing duck-type helper.** `_consume_new_stdout_lines` already encapsulates the `getattr`-check-and-call convention. Exporting it as `consume_provider_stdout_lines` gives custom adapters the same call site as the production adapter with zero new structural surface. Accepted.

## Consequences

- `RuntimeClient` gains `__init__(provider_invocation_adapter=None)`. The implicit no-argument constructor still works unchanged.
- All three run entry-points (`run_ephemeral`, `run_new_session`, `run_resumed_session`) thread `self._provider_invocation_adapter` into their internal `_run_builtin_*` calls. This also closes the previously untracked ephemeral-run gap where `RuntimeClient.run_ephemeral` did not thread through any adapter.
- Six symbols are added to `agent_runtime.__all__`: `ProviderInvocationAdapter`, `ProviderInvocationRequest`, `ProviderInvocationResult`, `ProviderInvocationFailure`, `InvocationFailureKind`, `consume_provider_stdout_lines`. Their `__module__` is set to `agent_runtime`.
- The internal `_provider_invocation` module that defines these types remains private and must not be imported by consumers directly.
- `ProviderUnavailableReason` (needed for `ProviderInvocationFailure.provider_unavailable_reason`) is already reachable from the non-underscore `agent_runtime.errors` module; it is not added to `__all__` here and is left as a separate surface decision.
