# Runtime ownership migration

Reusable runtime layer owns shared execution contracts and lifecycle behavior: service selection, provider selection, session planning, provider state contracts, work invocation, result/error vocabulary, importable runtime modules. Not the application — receives prepared inputs, returns results or runtime-owned failures.

## Decision

- Runtime package owns reusable execution contracts and lifecycle behavior.
- Narrow, adapter-driven public surface.
- Must import without application modules.
- Application orchestration outside runtime distribution.
- Provider-specific policy behind runtime adapter contracts.
- Built artifact matches editable-source boundary.

## Consequences

- Core behavior reusable without importing a consuming application.
- Application code stays an adapter around runtime-owned contracts.
- Boundary regressions surface as package import failures and artifact tests.
