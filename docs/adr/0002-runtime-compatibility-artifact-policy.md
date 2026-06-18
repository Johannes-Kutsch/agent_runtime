# Runtime compatibility artifact policy

Status: Partially superseded by [0009 - Built-in provider-only runtime](0009-built-in-provider-only-runtime.md) for provider dependencies shipped with built-in Claude, Codex, and OpenCode integrations.

The runtime package should expose neutral public vocabulary. Compatibility names, legacy paths, or older record shapes may exist only as adapter-layer shims when they are needed to preserve existing behavior.

## Decision

- Use neutral public names for runtime-owned errors, session paths, and log records.
- Require caller-supplied paths and directories where filesystem layout matters.
- Treat any compatibility-specific vocabulary as a transitional adapter concern.
- Keep supported Python version metadata and classifiers aligned with the versions verified for release.
- Keep the runtime package dependency-free at install time; provider-specific and development dependencies stay outside the runtime dependency set.

## Consequences

- The runtime package can be adopted without inheriting application-specific naming.
- Layout decisions stay at the adapter boundary instead of becoming hidden defaults.
- Tests can assert that neutral public contracts remain the default.
- Consumers can rely on package metadata to reflect the supported Python runtime versions.
- Consuming projects can adopt the runtime boundary without inheriting provider-specific dependency trees.
