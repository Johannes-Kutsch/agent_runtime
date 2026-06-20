# Runtime public surface narrowing

Status: Partially refined by [0005 - Runtime session lifecycle entrypoints](0005-runtime-session-lifecycle-entrypoints.md) and [0006 - Built-in provider-only runtime](0006-built-in-provider-only-runtime.md). Current glossary and public API docs own exact `Invocation Directory`, `ToolPolicy`, and transitional vocabulary.

The runtime boundary should expose a smaller, clearer front-facing surface: one canonical runtime entrypoint per mode, a narrow package root, and focused seams for work invocation, session planning, and provider policy.

## Decision

- Keep the package root as a narrow compatibility entrypoint, not a catch-all export surface.
- Keep `ruhken-agent-runtime` as distribution name and `agent_runtime` as import package name.
- Use package metadata for installed version lookup rather than package-root `__version__`.
- Keep core values such as `StageSelection`, `ToolPolicy`, `ProviderAuth`, `Continuation`, and outcome/result values public without promoting implementation modules.
- Keep behaviorful lifecycle entrypoints in focused runtime modules.
- Keep runtime execution entrypoints async-only until synchronous wrappers have a proven consumer need.
- Keep runtime import-isolation checks as internal self-test infrastructure.
- Validate runtime service identities used in registry, paths, logs, and diagnostics as path-safe service names.
- Treat model and effort as provider execution parameters, not path-safe runtime identities.
- Expose one canonical lifecycle entrypoint per mode instead of parallel facades and free functions.
- Give each canonical mode its own request type instead of aliasing low-level work or prompt request shapes.
- Keep prompt inputs as already-rendered text strings rather than structured message schemas.
- Require explicit tool policy rather than inheriting provider defaults.
- Keep public request, result, and metadata dataclasses immutable.
- Avoid untyped extension holes on public request/session objects unless replaced by named protocols or value types.
- Return normalized text and grouped runtime metadata from canonical results.
- Keep fallback diagnostics explicit where consumers commonly need them.
- Use plain invocation directories in canonical APIs unless a richer mount abstraction is actually configurable.
- Keep canonical requests focused on execution intent, not presentation or status UI wiring.
- Keep container workspace paths out of canonical APIs; container projection is low-level execution plumbing.
- Use selection vocabulary for stage/service/model/effort candidate chains instead of override vocabulary.
- Require public stage-selection nodes to provide explicit service, model, and effort values.
- Keep recursive fallback links as the public stage-selection chain shape.
- Keep service, model, and effort coupled per stage selection node.
- Keep work invocation dependencies focused on runtime execution rather than presentation or orchestration concerns.
- Keep provider event dataclasses as provider output contracts instead of untyped event envelopes.
- Preserve raw provider diagnostic observations in adapter contracts; consumers own display, storage, and redaction.
- Expose provider-output reduction helpers through focused adapter seams, not low-level work invocation modules.
- Keep invocation logging as an advanced focused seam: runtime owns record shape, consumers own location, persistence, retention, and presentation.
- Keep service selection and availability policy separate from presentation helpers.
- Keep `CancellationToken` as runtime-owned cooperative cancellation; keep quota/fallback bookkeeping separate from caller cancellation.
- Use service-name vocabulary for runtime-owned usage-limit identity.
- Keep usage-limit account labels diagnostic, not selection identity.
- Keep `provider_session_id` distinct from runtime service names.
- Keep `ToolPolicy` as the coarse runtime-owned execution policy enum.
- Keep intermediate tool policies because downstream integrations need more than no-tools/full-tools.
- Keep tool-policy command mappings behind provider adapters.
- Provider adapters own translation into provider-specific CLI flags and limitations.
- Keep `RunKind` as a closed runtime-owned session lifecycle enum.
- Use `RunKind.RESUME` for provider-session resume lifecycle while public mode names use session lifecycle vocabulary.
- Treat low-level work invocation modules as implementation modules, even when importable for compatibility and tests.
- Treat direct `invoke_work` usage as undocumented implementation API.

## Consequences

- The runtime boundary is easier to learn and test through its public surface.
- The published distribution name carries the intended package namespace while import stays concise.
- Package-root imports remain small vocabulary, not a full runtime facade.
- Boundary self-tests remain internal.
- Consumers can import core values directly without learning implementation modules.
- Selection and quota availability policy can evolve without widening execution contracts.
- Service names stay safe identity keys across selection, state layout, logs, and diagnostics.
- Provider adapters retain provider-specific model and effort validation.
- Service availability summaries can vary by consuming application without runtime presentation APIs.
- Callers have one ordinary way to reach each behavior.
- Mode-specific request types can evolve without freezing implementation shapes into consumer API.
- Application prompt rendering remains outside the runtime boundary.
- Boundary values are safe to pass through async execution without accidental mutation.
- Release surface avoids UI/status concerns even if implementation modules still contain presentation plumbing.
- Ordinary consumers do not need container workspace mechanics.
- Consumers can describe ordered runtime candidates without implying hidden override state.
- Missing runtime candidate configuration fails at construction or validation boundaries.
- Ordered fallback remains expressible without a separate list or graph abstraction.
- Fallback services can carry provider-appropriate model and effort settings.
- Release work can prioritize canonical consumer seams without hiding every implementation module first.
- Ordinary consumers should use runtime entrypoints and adapter seams instead of work invocation internals.
- Provider output remains type-directed while staying behind adapter contracts.
- Provider failure diagnostics remain useful without runtime pretending to sanitize provider payloads.
- Shared log lifecycle remains reusable without making logging configuration ordinary request model.
- Long-running invocations remain cancellable without exposing lower-level async primitives.
- Usage-limit policy refers to runtime service identity.
- Runtime service identity and external provider session identity remain separate.
- Tool-policy values stay small while preserving downstream-required intermediate access.
- Shared policy helpers can reduce duplicated adapter logic without moving provider CLI syntax into public runtime API.
