# Live provider smoke runner

Status: Accepted.

Deterministic tests cover lifecycle behavior, command rendering, tool policy, live output observation, and portable continuations. They do not prove real Claude, Codex, and OpenCode integrations can invoke provider commands with host credentials and live provider availability.

The project needs an opt-in Live Provider Smoke Test for maintainers and credentialed CI. It must prove the Runtime Public Surface contract without becoming a provider benchmark, default suite dependency, or runtime public API.

## Decision

- Add an opt-in standalone live smoke runner as maintainer tooling under `scripts/`.
- Keep the runner out of default pytest and installed Runtime Public Surface.
- Use only public runtime imports and request values: `RuntimeClient`, lifecycle request values, `StageSelection`, `ProviderAuth`, `Continuation`, and public `ToolPolicy`.
- Treat `RuntimeStateDir`, `RuntimeLogsDir`, `SessionNamespace`, `ToolAccess`, and `ToolPolicyProfile` as transitional or internal vocabulary for this tooling purpose.
- Make provider selection explicit: `claude`, `codex`, `opencode`, multiple explicit providers, or `all`.
- In explicit provider selection, missing provider config is a configuration error; in `all`, skip unconfigured providers but fail when none are configured.
- Resolve provider model and effort from provider-specific environment variables and CLI flags, with CLI flags taking precedence.
- Support separate lifecycle and tool-policy smoke modes, plus a combined mode.
- Lifecycle smoke proves invocation health and session continuity with non-empty outputs, meaningful continuation, and resumed output containing the prior sentinel without requiring exact provider output.
- Capture Live Runtime Output when available, but absence of live turns must not fail the smoke run.
- Tool-policy smoke uses Ephemeral Run only and proves successful invocation under each public `ToolPolicy`, not tool usefulness, sandbox enforcement, or mutation behavior.
- Do not test fallback chains, cancellation, every model/effort combination, custom tool-policy profiles, or raw provider streams.
- Run provider cases serially; in combined mode group by provider, run lifecycle first, and skip a provider's tool-policy matrix if lifecycle failed.
- Provide dry-run and provider-listing modes that validate selection and credential presence without invoking providers.
- Preserve artifacts by default under a repo-local gitignored artifact root, with override and explicit cleanup options.
- Use one run id per smoke invocation, allow path-safe user-supplied run ids, and derive per-case sentinels from the run id.
- Write a JSON summary artifact for real runs, optionally emit JSON to stdout, and use zero/non-zero process exit only.
- Preserve runner-owned diagnostics: config summaries, prompts, final outputs, live turns, outcome summaries, invocation records when returned, timings, and tracebacks.
- Do not preserve credentials, raw environment dumps, auth files, credential-derived values, or opaque home-directory provider state.
- Do not automatically redact provider output; mark artifacts as potentially sensitive and keep them under the local ignored artifact root.
- Keep script help as documentation surface unless tooling becomes a broader maintained workflow.

## Consequences

- Maintainers can verify real built-in provider invocation without making normal tests depend on live services, credentials, provider quota, or network availability.
- The runner exercises the same public lifecycle entrypoints ordinary consumers use.
- Live smoke results identify invocation, continuation, configuration, availability, and artifact failures, but do not certify answer quality or tool behavior.
- CI can opt into protected credentialed smoke jobs with stable JSON output.
- Runner-created artifacts may be sensitive; implementation must keep artifact root ignored and avoid credential capture.
