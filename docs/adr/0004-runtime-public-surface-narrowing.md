# Runtime public surface narrowing

Status: Refined by [0005 - Runtime session lifecycle entrypoints](0005-runtime-session-lifecycle-entrypoints.md), [0006 - Built-in provider-only runtime](0006-built-in-provider-only-runtime.md), and [0010 - Single-candidate provider selection](0010-single-candidate-provider-selection.md). ADR 0010 is the current provider-selection decision; current glossary and public API docs own exact `Invocation Directory`, `ToolPolicy`, and `ProviderSelection` vocabulary.

The runtime boundary exposes a small, clear front-facing surface: one canonical entrypoint per mode, a narrow package root, and focused seams for work invocation, session planning, and provider policy.

## Decision

- Package root is a narrow compatibility entrypoint, not a catch-all export surface.
- `ruhken-agent-runtime` distribution name, `agent_runtime` import package name; installed version via package metadata, not package-root `__version__`.
- Keep core values (`ToolPolicy`, `ProviderAuth`, `Continuation`, `ProviderSelection`, outcome/result values) public without promoting implementation modules.
- Behaviorful lifecycle entrypoints live in focused runtime modules: one canonical entrypoint per mode, not parallel facades or free functions. Each mode gets its own request type, not aliased low-level work/prompt shapes.
- Execution entrypoints async-only until synchronous wrappers have proven need.
- Import-isolation checks stay internal self-test infrastructure.
- Validate runtime service identities (registry, paths, logs, diagnostics) as path-safe names; model and effort are provider execution parameters, not path-safe identities.
- Prompt inputs are already-rendered text strings, not structured message schemas.
- Require explicit tool policy, not provider defaults.
- Public request/result/metadata dataclasses immutable; no untyped extension holes (use named protocols/value types).
- Canonical results return normalized text and grouped runtime metadata; keep selected-provider diagnostics explicit.
- Plain invocation directories in canonical APIs unless a richer mount abstraction is configurable; keep container workspace paths out (low-level plumbing). Canonical requests focus on execution intent, not presentation/status UI.
- Provider-selection shape and fallback ownership superseded here by [0010 - Single-candidate provider selection](0010-single-candidate-provider-selection.md): one `ProviderSelection` per invocation, fallback owned by consumers across separate invocations.
- Provider event dataclasses are provider output contracts, not untyped event envelopes; preserve raw provider diagnostics in adapter contracts (consumers own display/storage/redaction). Provider-output reduction helpers via focused adapter seams, not low-level work invocation modules.
- Invocation logging is an advanced focused seam: runtime owns record shape, consumers own location/persistence/retention/presentation.
- Service selection/availability policy separate from presentation helpers.
- `CancellationToken` is runtime-owned cooperative cancellation; quota/fallback bookkeeping separate from caller cancellation.
- Service-name vocabulary for usage-limit identity; usage-limit account labels are diagnostic, not selection identity. `provider_session_id` distinct from runtime service names.
- `ToolPolicy` is the coarse runtime-owned execution policy enum; keep intermediate policies (downstream needs more than none/full); tool-policy command mappings behind provider adapters.
- `RunKind` is a closed runtime-owned session lifecycle enum; `RunKind.RESUME` for provider-session resume while public mode names use session lifecycle vocabulary.
- Low-level work invocation modules are implementation modules even when importable; direct `invoke_work` usage is undocumented implementation API.

## Consequences

- Boundary easier to learn and test through its public surface; consumers import core values without learning implementation modules.
- Provider-selection and quota policy can evolve without widening execution contracts; consumer-owned fallback chooses provider-appropriate model/effort across invocations.
- Service names stay safe identity keys across selection, state layout, logs, diagnostics; runtime and external provider session identity stay separate.
- Provider failure diagnostics stay useful without runtime pretending to sanitize payloads.
- Long-running invocations stay cancellable without exposing lower-level async primitives.
