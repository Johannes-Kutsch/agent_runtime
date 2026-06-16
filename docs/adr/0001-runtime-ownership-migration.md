# Runtime ownership migration

The reusable runtime layer owns the contracts and behavior that can be shared by multiple consuming projects: service selection, stage-chain resolution, session planning, provider state contracts, work invocation, and agent log lifecycle.

The runtime package is not the application. It receives already-prepared inputs from an adapter boundary and returns execution results and runtime-owned errors. Application-specific prompt rendering, command wiring, issue orchestration, and UI belong outside this package.

## Decision

- Make the runtime package the owner of reusable execution contracts and lifecycle behavior.
- Keep the public surface narrow and adapter-driven.
- Require runtime importability without application modules.

## Consequences

- Core behavior can be reused without importing a consuming application.
- Application code becomes a thin adapter around runtime-owned contracts.
- Package-level tests can prove the boundary independently.
