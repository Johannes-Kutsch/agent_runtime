# Live provider smoke runner

Status: Accepted.

The runtime has deterministic automated tests for lifecycle behavior, provider command rendering, tool policy handling, live output observation, and portable continuations. Those tests do not prove that the real built-in Claude, Codex, and OpenCode integrations can currently invoke provider commands with real host credentials and live provider availability.

The project needs an opt-in Live Provider Smoke Test for maintainers and CI jobs with credentials. The smoke runner must prove the documented Runtime Public Surface contract without becoming a provider quality benchmark, a default test-suite dependency, or another runtime public API surface.

## Decision

- Add an opt-in standalone live smoke runner as maintainer tooling under `scripts/`.
- Keep the runner out of the default pytest suite and out of the installed Runtime Public Surface.
- Use only public runtime imports and public request values: `RuntimeClient`, lifecycle request values, `StageSelection`, `ProviderAuth`, `Continuation`, and public `ToolPolicy` values.
- Treat `RuntimeStateDir`, `RuntimeLogsDir`, `SessionNamespace`, `ToolAccess`, and `ToolPolicyProfile` as transitional or internal vocabulary for this purpose; the smoke runner should follow the documented ordinary consumer request shape.
- Make provider selection explicit and runtime-vocabulary-aligned: `claude`, `codex`, `opencode`, multiple explicit providers, or `all`.
- In explicit provider selection, missing provider config is a configuration error. In `all`, unconfigured providers are skipped, but an `all` run with zero configured providers is a configuration error.
- Resolve provider model and effort from provider-specific environment variables and CLI flags, with CLI flags taking precedence. Do not hard-code live model or effort defaults unless the public docs establish such defaults.
- Support separate lifecycle and tool-policy smoke modes, plus a combined mode.
- Lifecycle smoke should prove invocation health and session continuity: Ephemeral Run completes with non-empty output, Start Session Run completes with non-empty output and a meaningful continuation, and Resume Session Run completes with non-empty output containing the sentinel from the previous session turn.
- Lifecycle smoke should not require exact provider output or fail on extra provider text.
- Capture Live Runtime Output observations when available, but do not make their absence fail the smoke run; completed `RuntimeOutcome` output remains authoritative.
- Tool-policy smoke should use Ephemeral Run only and cover `ToolPolicy.NONE`, `ToolPolicy.INSPECT_ONLY`, `ToolPolicy.NO_FILE_MUTATION`, and `ToolPolicy.UNRESTRICTED`.
- Tool-policy smoke should prove successful invocation under each policy, not provider tool usefulness, sandbox enforcement, or file mutation behavior.
- Do not test provider fallback chains, cancellation behavior, every supported model/effort combination, custom tool-policy profiles, or raw provider stream behavior in this runner.
- Run provider cases serially. In combined mode, group by provider and run lifecycle before tool-policy cases; skip a provider's tool-policy matrix when that provider's lifecycle smoke has already failed.
- Provide dry-run and provider-listing modes that validate selection and credential presence without invoking providers.
- Preserve artifacts by default under a repo-local gitignored artifact root, with an override and explicit cleanup option.
- Use one run id per smoke invocation, allow path-safe user-supplied run ids, and derive per-case sentinels from the run id.
- Write a JSON summary artifact for real runs, optionally emit JSON to stdout, and use zero/non-zero process exit only. Detailed status categories belong in the JSON summary.
- Preserve runner-owned diagnostics such as config summaries, prompts, final outputs, live turns, outcome summaries, invocation records when returned, timings, and tracebacks.
- Do not preserve credentials, raw environment dumps, auth files, credential-derived values, or opaque home-directory provider state.
- Do not automatically redact provider output; mark artifacts as potentially sensitive and keep them under the local ignored artifact root.
- Keep script help as the documentation surface for the runner. Do not document it in the README or public API docs unless the tooling becomes a broader maintained workflow.

## Consequences

- Maintainers can verify real built-in provider invocation without making the normal test suite depend on live services, credentials, provider quota, or network availability.
- The smoke runner exercises the same public lifecycle entrypoints that ordinary consumers use, so failures are meaningful at the runtime boundary.
- Live smoke results remain intentionally coarse: they identify invocation, continuation, configuration, availability, and artifact failures, but they do not certify provider answer quality or tool behavior.
- CI can opt into protected credentialed smoke jobs with stable JSON output while local developers can run focused provider or mode subsets.
- The runner will create sensitive diagnostic artifacts by default, so the implementation must keep the artifact root ignored and avoid credential capture.
