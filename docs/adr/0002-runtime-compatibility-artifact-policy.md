# Runtime compatibility artifact policy

Status: Partially refined by [0006 - Built-in provider-only runtime](0006-built-in-provider-only-runtime.md) for built-in Claude, Codex, and OpenCode integration behavior.

The runtime package should expose neutral public vocabulary. Compatibility names, legacy paths, or older record shapes may exist only as adapter-layer shims when needed to preserve existing behavior.

## Decision

- Use neutral public names for runtime-owned errors, session paths, and log records.
- Require caller-supplied paths and directories where filesystem layout matters.
- Treat compatibility-specific vocabulary as a transitional adapter concern.
- Keep supported Python version metadata and classifiers aligned with verified release versions.
- Do not let provider migration artifacts define public runtime vocabulary.
- Keep install-time dependencies absent unless a built-in provider dependency is deliberately shipped as runtime behavior; development dependencies stay outside the runtime dependency set.

## Consequences

- The runtime package can be adopted without inheriting application-specific naming.
- Layout decisions stay at the adapter boundary instead of becoming hidden defaults.
- Tests can assert that neutral public contracts remain the default.
- Consumers can rely on package metadata to reflect supported Python versions.
- Consumers do not inherit provider migration or development dependency trees accidentally.
