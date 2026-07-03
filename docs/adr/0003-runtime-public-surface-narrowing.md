# Runtime public surface narrowing

Status: Refined by [0004 - Runtime session lifecycle entrypoints](0004-runtime-session-lifecycle-entrypoints.md), [0005 - Built-in provider-only runtime](0005-built-in-provider-only-runtime.md), [0008 - Single-candidate provider selection](0008-single-candidate-provider-selection.md), and [0020 - Wiring CancellationToken into invocation execution](0020-cancellation-token-wiring.md). ADR 0008 is the current provider-selection decision; ADR 0020 is the current `CancellationToken` decision.

Small, clear front-facing surface: one canonical entrypoint per mode, narrow package root, focused seams for work invocation, session planning, provider policy.

## Decision

- Package root is narrow compatibility entrypoint, not catch-all export surface.
- `ruhken-agent-runtime` distribution, `agent_runtime` import; version via package metadata, not `__version__`.
- Core values (`ToolPolicy`, `ProviderAuth`, `Continuation`, `ProviderSelection`, outcome/result) public without promoting implementation modules.
- One canonical lifecycle entrypoint per mode in focused runtime modules, not parallel facades. Each mode gets its own request type.
- Async-only execution entrypoints until sync wrappers have proven need.
- Import-isolation checks stay internal self-test infrastructure.
- Service identities validated as path-safe names; model/effort are execution parameters, not path-safe identities.
- Prompt inputs are already-rendered text strings, not structured message schemas.
- Explicit tool policy required, not provider defaults.
- Public request/result/metadata dataclasses immutable; no untyped extension holes.
- Plain invocation directories; container workspace paths are low-level plumbing.
- Provider-selection: one `ProviderSelection` per invocation, consumer-owned fallback across invocations (see [0008](0008-single-candidate-provider-selection.md)).
- Provider event dataclasses are output contracts, not untyped envelopes; raw diagnostics preserved, consumers own display/storage/redaction.
- Invocation logging: runtime owns record shape, consumers own persistence/retention/presentation.
- `CancellationToken` is runtime-owned cooperative cancellation; separate from quota/fallback bookkeeping.
- Service-name vocabulary for usage-limit identity; `provider_session_id` distinct from runtime service names.
- `ToolPolicy`: coarse runtime-owned enum with intermediate policies; command mappings behind provider adapters.
- `RunKind`: closed lifecycle enum; `RunKind.RESUME` for provider-session resume, public names use session lifecycle vocabulary.
- Low-level work invocation modules are undocumented implementation API even when importable.

## Consequences

- Boundary learnable through public surface; consumers import core values without learning internals.
- Selection and quota policy evolve without widening execution contracts.
- Service names stay safe identity keys; runtime and provider session identity stay separate.
- Provider failure diagnostics stay useful without runtime sanitizing payloads.
- Long-running invocations cancellable without exposing lower-level async primitives.
