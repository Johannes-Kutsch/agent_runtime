# Runtime ownership migration

The reusable runtime layer owns shared execution contracts and lifecycle behavior: service selection, provider selection, session planning, provider state contracts, work invocation, result/error vocabulary, and importable runtime modules.

The runtime package is not the application. It receives prepared inputs from an adapter boundary and returns execution results or runtime-owned failures. Application prompt rendering, command wiring, issue orchestration, UI, and presentation stay outside the package.

## Decision

- Runtime package owns reusable execution contracts and lifecycle behavior.
- Keep the public surface narrow and adapter-driven.
- Runtime must import without application modules.
- Keep application orchestration out of the runtime distribution.
- Keep provider-specific policy behind runtime adapter contracts.
- Built artifact must match the editable-source boundary.

## Consequences

- Core behavior reusable without importing a consuming application.
- Application code stays an adapter around runtime-owned contracts.
- Boundary regressions surface as package import failures and artifact tests.
