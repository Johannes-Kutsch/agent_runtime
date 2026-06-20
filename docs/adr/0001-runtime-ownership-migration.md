# Runtime ownership migration

The reusable runtime layer owns shared execution contracts and lifecycle behavior: service selection, stage-chain resolution, session planning, provider state contracts, work invocation, result/error vocabulary, and importable runtime modules.

The runtime package is not the application. It receives prepared inputs from an adapter boundary and returns execution results or runtime-owned failures. Application prompt rendering, command wiring, issue orchestration, UI, and presentation stay outside the package.

## Decision

- Make the runtime package the owner of reusable execution contracts and lifecycle behavior.
- Keep the public surface narrow and adapter-driven.
- Require runtime importability without application modules.
- Keep application orchestration out of the runtime distribution.
- Keep provider-specific policy behind runtime adapter contracts.
- Require the built artifact to match the editable-source boundary.

## Consequences

- Core behavior can be reused without importing a consuming application.
- Application code remains an adapter around runtime-owned contracts.
- Boundary regressions show up as package import failures and artifact tests.
