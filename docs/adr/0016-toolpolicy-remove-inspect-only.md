# `INSPECT_ONLY` retired; ToolPolicy narrows to NONE / NO_FILE_MUTATION / UNRESTRICTED

`ToolPolicy` had four values: `NONE`, `INSPECT_ONLY`, `NO_FILE_MUTATION`, `UNRESTRICTED`. The distinction between `INSPECT_ONLY` and `NONE` was not worth maintaining.

`INSPECT_ONLY` was an allowlist policy: Claude received `--tools "Read Glob"` only; OpenCode had both `edit` and `bash` denied. Without bash access an agent cannot find files, run commands, or do meaningful work — making it functionally indistinguishable from `NONE` in practice. For Codex, `INSPECT_ONLY` and `NONE` already produced identical rendering (`--sandbox read-only`); the provider could not tell them apart at all.

## Decision

- Delete `INSPECT_ONLY` from the Runtime Public Surface. No compatibility alias.
- `ToolPolicy` becomes a three-value closed enum: `NONE`, `NO_FILE_MUTATION`, `UNRESTRICTED`.
- Consumers needing "no file mutations, bash permitted" use `NO_FILE_MUTATION`. Consumers needing "no tools at all" use `NONE`.

## Rationale

Retaining `INSPECT_ONLY` would require a corresponding container-mode variant for sandboxed execution environments, compounding the surface without meaningful behavioral gain. The policy's defining property — allowlisting Read and Glob while denying bash — renders the agent unable to perform most tasks, collapsing it onto `NONE` in practice. Removing it shrinks the public surface and removes an invisible no-op distinction at the Codex layer.

We reject aliasing `INSPECT_ONLY` to `NONE` or `NO_FILE_MUTATION` as a compatibility shim. Runtime Compatibility Aliases are not Runtime Public Surface promises; a clean deletion is consistent with the pattern established in ADRs 0003, 0012, and the session-seam retirements.

## Consequences

- Runtime Public Surface loses one `ToolPolicy` value; existing consumers using `INSPECT_ONLY` must migrate to `NONE` or `NO_FILE_MUTATION`.
- Codex rendering for `NONE` and `NO_FILE_MUTATION` remains `--sandbox read-only`; no rendering changes required.
- The `ToolPolicyProfile` allowlist path (`allowed_tools` non-None) becomes unreachable through the standard `ToolPolicy` enum; it survives as a low-level internal path only.
