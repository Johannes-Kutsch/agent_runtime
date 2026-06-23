# Runtime compatibility artifact policy

Status: Partially refined by [0006 - Built-in provider-only runtime](0006-built-in-provider-only-runtime.md) for built-in Claude, Codex, and OpenCode integration behavior.

Runtime exposes neutral public vocabulary. Compatibility names, legacy paths, or older record shapes exist only as adapter-layer shims when needed to preserve behavior.

## Decision

- Neutral public names for runtime-owned errors, session paths, and log records.
- Require caller-supplied paths/directories where filesystem layout matters.
- Compatibility-specific vocabulary is a transitional adapter concern.
- Keep supported Python version metadata/classifiers aligned with verified release versions.
- Provider migration artifacts must not define public runtime vocabulary.
- No install-time dependencies unless a built-in provider dependency is deliberately shipped as runtime behavior; dev dependencies stay outside the runtime dependency set.

## Consequences

- Package adoptable without inheriting application-specific naming.
- Layout decisions stay at the adapter boundary, not hidden defaults.
- Tests can assert neutral public contracts remain default.
- Consumers don't inherit provider-migration or dev dependency trees accidentally.
