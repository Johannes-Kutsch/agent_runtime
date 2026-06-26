# Already-Sandboxed Execution: `RuntimeClient` flag skips Codex OS sandbox when host provides isolation

Codex uses `--sandbox read-only` for `NONE` and `NO_FILE_MUTATION` `ToolPolicy` values. This instructs Codex to set up an OS-level sandbox using bubblewrap (Linux), Seatbelt (macOS), or the Windows sandbox before executing. Inside a standard Docker container, that sandbox-creation step fails: user namespaces are unavailable without `--privileged`, and bubblewrap may not be installed. Codex crashes at startup before doing any work.

`--sandbox danger-full-access` skips sandbox creation entirely, allowing Codex to start. The official Codex recommendation for container environments is to use `danger-full-access` and let the container itself be the hard isolation boundary. Claude and OpenCode are unaffected by this problem — both use logical/config-level restrictions (`--tools`/`--disallowedTools` and `OPENCODE_CONFIG_CONTENT` respectively) that work in any execution environment.

Downstream consumers running agent invocations inside Docker containers need a way to signal this to the runtime. Using `ToolPolicy.UNRESTRICTED` as a workaround is incorrect: it simultaneously removes Claude's `--disallowedTools` and OpenCode's permission config, stripping behavioral guidance from providers that don't need the workaround.

## Decision

- Add `already_sandboxed: bool = False` to the `RuntimeClient` constructor. This is **Already-Sandboxed Execution** configuration.
- When `already_sandboxed=True`: Built-in Provider Rendering passes `--sandbox danger-full-access` to Codex regardless of `ToolPolicy`. Claude and OpenCode rendering are unchanged.
- `ToolPolicy` continues to shape agent behavior for all providers. For Codex in Already-Sandboxed Execution, `ToolPolicy` enforcement is behavioral (container as hard boundary) rather than OS-enforced.
- `already_sandboxed` lives at the `RuntimeClient` constructor, not on individual run request types.

## Rationale

"Running in a container" is a deployment-time constant for a given process, not a per-invocation decision. Placing the flag on run request types (`start_session_run`, `resume_session_run`, `ephemeral_run`) would create per-call noise and an accidental-inconsistency hazard: a Start Session Run with the flag followed by a Resume Session Run without it would render different Codex sandbox arguments for the same logical session.

We reject introducing new `ToolPolicy` variants (e.g. `NO_FILE_MUTATION_CONTAINER`) to encode the container context. `ToolPolicy` is a capability grant — what the agent may do — not an execution environment descriptor. Conflating the two would double the policy surface, leak deployment context into capability semantics, and require a new variant for every future provider that encounters the same OS-sandbox-in-container problem.

We reject per-run flags for the same reasons: the deployment environment does not change between runs on the same client.

## Consequences

- Downstream consumers in Docker containers can use any `ToolPolicy` with Codex without encountering sandbox-creation failures.
- For Codex under Already-Sandboxed Execution, `ToolPolicy` behavioral shaping is best-effort: the container is the hard boundary, not Codex's OS sandbox.
- `RuntimeClient` gains one constructor parameter on the Runtime Public Surface; all existing call sites default to `already_sandboxed=False`, preserving current behavior.
- Claude and OpenCode behavior is unaffected by `already_sandboxed`.
