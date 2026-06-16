# Runtime compatibility artifact policy

The runtime package should expose neutral public vocabulary. Compatibility names, legacy paths, or older record shapes may exist only as adapter-layer shims when they are needed to preserve existing behavior.

## Decision

- Use neutral public names for runtime-owned errors, session paths, and log records.
- Require caller-supplied paths and directories where filesystem layout matters.
- Treat any compatibility-specific vocabulary as a transitional adapter concern.

## Consequences

- The runtime package can be adopted without inheriting application-specific naming.
- Layout decisions stay at the adapter boundary instead of becoming hidden defaults.
- Tests can assert that neutral public contracts remain the default.
