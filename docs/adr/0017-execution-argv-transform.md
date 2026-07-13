# Execution Argv Transform replaces Already-Sandboxed Execution

Consumers need to route provider CLI execution into Docker containers without ar gaining a Docker dependency. The prior solution (`already_sandboxed` on the `RuntimeClient` constructor) only addressed Codex's OS sandbox and was constructor-scoped, which breaks when a single client dispatches invocations to different containers.

We add an optional `Execution Argv Transform` callable to all three run request types (`EphemeralRunRequest`, `NewSessionRunRequest`, `ResumedSessionRunRequest`). ar applies it to the fully-rendered canonical argv, Invocation Directory, and rendered environment before executing, then runs the returned argv via its existing subprocess machinery. ar retains full subprocess ownership (stdin, stdout/stderr, idle timeout, exit-code handling); the consumer provides a pure synchronous data transformation only.

Per-request placement is intentional: a single `RuntimeClient` may dispatch to multiple containers, so constructor-level scoping is too coarse.

When `Execution Argv Transform` is present on a Codex invocation, Built-in Provider Rendering automatically applies `--sandbox danger-full-access`. A custom transform implies a non-standard execution environment where Codex's OS sandbox cannot be assumed to work — triggered per-invocation rather than by a constructor flag.

`already_sandboxed` is removed from the `RuntimeClient` constructor. Its only use case is covered by the per-invocation transform.

## Considered Options

- **Subprocess launcher callable** (`(argv, cwd, env) → AsyncIterable[str]`): consumer owns the full launch and output stream. Rejected: ar loses subprocess ownership (timeout, stdin, exit-code logic moves to the consumer); async/sync mismatch with the current synchronous stack; consumer can intercept or drop the output stream.
- **Process factory** (`(argv, cwd, env) → Popen`): consumer creates the process object, ar drives it. Rejected: consumer must set `stdout=PIPE, stderr=PIPE, text=True` correctly or ar's stream reading silently breaks — a hidden contract with no enforcement.
- **Static argv prefix tuple**: too limited; no access to `cwd` or `env`, preventing `--workdir` and env injection.

## Consequences

- The rendered environment (`env`) is passed to the transform so consumers can inject ar-generated values — notably `OPENCODE_CONFIG_CONTENT`, a JSON provider-config string ar generates for OpenCode Go that OpenCode cannot receive via CLI flags and the consumer cannot reconstruct without ar internals.
- When a transform is present, Built-in Provider Invocation forces stdin prompt transport and applies the transform before host executable resolution (`shutil.which`), so host-resolved paths do not appear in the argv the consumer receives.
