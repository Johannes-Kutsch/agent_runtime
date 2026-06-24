# Runtime compatibility artifact policy

Status: Partially refined by [0005 - Built-in provider-only runtime](0005-built-in-provider-only-runtime.md) for built-in Claude, Codex, and OpenCode integration behavior.

Runtime exposes neutral public vocabulary. Compatibility names and legacy paths exist only as adapter-layer shims when needed.

## Decision

- Neutral public names for runtime-owned errors, session paths, log records.
- Caller-supplied paths/directories where filesystem layout matters.
- Compatibility-specific vocabulary is transitional adapter concern.
- Supported Python version metadata aligned with verified release versions.
- Provider migration artifacts must not define public runtime vocabulary.
- No install-time dependencies unless deliberately shipped as runtime behavior; dev dependencies stay outside.

## Consequences

- Package adoptable without inheriting application-specific naming.
- Layout decisions stay at adapter boundary.
- Tests can assert neutral public contracts remain default.
- Consumers don't inherit provider-migration or dev dependency trees.
