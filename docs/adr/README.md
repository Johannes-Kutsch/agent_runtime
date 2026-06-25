# Architecture Decision Records

- [0001 - Runtime ownership migration](0001-runtime-ownership-migration.md): Establish `agent_runtime` as reusable package boundary for execution contracts, session planning, service selection, and lifecycle behavior.
- [0002 - Runtime compatibility artifact policy](0002-runtime-compatibility-artifact-policy.md): Neutral public names; compatibility artifacts are adapter concerns.
- [0003 - Runtime public surface narrowing](0003-runtime-public-surface-narrowing.md): Narrow package root, focused seams, one canonical entrypoint per mode.
- [0004 - Runtime session lifecycle entrypoints](0004-runtime-session-lifecycle-entrypoints.md): Ephemeral, new-session, resumed-session entrypoints with portable consumer-owned continuations.
- [0005 - Built-in provider-only runtime](0005-built-in-provider-only-runtime.md): Ship Claude, Codex, OpenCode integrations; no consumer-defined provider services.
- [0006 - Provider invocation argv and stdin boundary](0006-provider-invocation-argv-stdin-boundary.md): Provider invocation as executable + arguments + prompt input, not shell strings.
- [0007 - No documentation regression tests](0007-no-documentation-regression-tests.md): Default tests assert executable behavior, not documentation wording.
- [0008 - Single-candidate provider selection](0008-single-candidate-provider-selection.md): One `ProviderSelection` per invocation; consumer-owned fallback across invocations.
- [0009 - Agent Event observation model](0009-agent-event-observation-model.md): Typed `Agent Event`s (message / tool call / other) via live feed only; `RuntimeOutcome` as `{kind, RunResult}` with `ResolvedProvider`.
- [0010 - Live Provider Probe: manual-debug-only](0010-live-provider-probe-manual-debug-only.md): Manual debugging tool exercising real providers; not CI, not Runtime Public Surface.
- [0011 - Merge provider stderr into observed stream](0011-merge-provider-stderr-into-observed-stream.md): Merge provider stderr into observed/reduced output stream so stderr-only failures reach the live feed.
- [0012 - Provider failure classification and outcome consolidation](0012-provider-failure-classification-and-outcome-consolidation.md): Consolidate `RetryableProviderFailure`/`NoServiceAvailable` into `ProviderUnavailable`; drop `ProviderErrorObservation`.
- [0013 - Silent provider hang timeout](0013-silent-provider-hang-timeout.md): Silent provider hangs stay honest `timed_out`; Idle Timeout moves into Built-in Provider Invocation and terminates the subprocess; event-layer watchdog retired.
- [0014 - Windows provider host environment allowlist](0014-windows-provider-host-env-allowlist.md): Restore a five-key Windows host-env allowlist (not a full merge), layered once for every built-in provider; Windows-only, no-op on POSIX.
