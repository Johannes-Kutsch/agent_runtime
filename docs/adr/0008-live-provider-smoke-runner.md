# Live provider smoke runner

Status: Accepted.

Deterministic tests cover lifecycle behavior, command rendering, tool policy, live output observation, and portable continuations. They do not prove real Claude, Codex, and OpenCode integrations can invoke provider commands with host credentials and live provider availability.

The project needs an opt-in Live Provider Smoke Test for maintainers and credentialed CI. It must prove the Runtime Public Surface contract without becoming a provider benchmark, default suite dependency, or runtime public API.

## Decision

- Add an opt-in standalone live smoke runner as maintainer tooling under `scripts/`.
- Keep live provider execution out of default pytest and installed Runtime Public Surface.
- Allow default pytest to cover smoke-runner planning and artifact behavior only when provider auth, provider availability, paths, and invocation behavior are injected or faked.
- Use only public runtime imports and request values: `RuntimeClient`, lifecycle request values, `ProviderSelection`, `ProviderAuth`, `Continuation`, and public `ToolPolicy`.
- Treat `RuntimeStateDir`, `RuntimeLogsDir`, `SessionNamespace`, `ToolAccess`, and `ToolPolicyProfile` as internal vocabulary, not smoke-runner public surface.
- Make provider selection explicit: `claude`, `codex`, `opencode`, multiple explicit providers, or `all`.
- In explicit provider selection, missing provider config is a configuration error; in `all`, skip unconfigured providers but fail when none are configured.
- Friendly provider-specific smoke commands keep explicit-provider semantics; friendly all-configured smoke commands keep `all` semantics.
- Resolve provider model and effort with precedence `CLI override > hardcoded smoke default`; live smoke does not use shell environment variables for configuration.
- Choose hardcoded smoke defaults from runtime-supported provider/model/effort tuples, not provider-global model catalogs.
- Prefer the cheapest runtime-supported model and lowest supported effort for each provider; maintainers can override when they want stronger models.
- Use initial hardcoded defaults `claude=haiku/low`, `codex=gpt-5.4-mini/low`, and `opencode=deepseek-v4-flash/medium`, subject to verification against provider availability at implementation time.
- Keep provider credential/configuration semantics unchanged: defaults only fill missing model and effort values.
- Load provider credentials for live smoke only from fixed `scripts/live-smoke/.env`, with `scripts/live-smoke/.env.example` as the committed template; do not use shell environment variables for live-smoke credentials and do not load `.env` from `RuntimeClient` or runtime public lifecycle entrypoints.
- Treat absent `.env` as valid for all-configured runs when only host-auth providers such as Codex are configured; treat missing, blank, or whitespace explicit-credential values as missing credentials.
- Live smoke tooling reads `.env` but never writes or mutates local credential files.
- Parse `.env` with a deliberately small grammar: non-empty lines must be simple `KEY=value`, surrounding whitespace is trimmed, and malformed lines fail fast with line numbers.
- Document the selected default tuples and verification date in tests or nearby maintainer help; do not preserve detailed pricing rationale as durable project text.
- Support separate lifecycle and tool-policy smoke modes, plus a combined mode.
- Lifecycle smoke proves invocation health and session continuity with non-empty outputs, meaningful continuation, and resumed output containing the prior sentinel without requiring exact provider output.
- Capture Live Runtime Output when available, but absence of live turns must not fail the smoke run.
- Tool-policy smoke uses Ephemeral Run only and proves successful invocation under each public `ToolPolicy`, not tool usefulness, sandbox enforcement, or mutation behavior.
- Do not test consumer fallback orchestration, cancellation, every model/effort combination, custom tool-policy profiles, or raw provider streams.
- Run provider cases serially; in combined mode group by provider, run lifecycle first, and skip a provider's tool-policy matrix if lifecycle failed.
- Do not provide dry-run or provider-listing modes; live smoke commands execute providers directly because the expected smoke cost is negligible, and missing configuration is reported at run start.
- Provide friendly full-matrix subcommands: `run` for all configured providers and `run <provider>` for one explicit provider; keep lower-level flags for targeted debugging.
- A provider selection without targeting flags means the Full Live Smoke Matrix; lifecycle or policy flags narrow the run for targeted reruns.
- Targeted rerun flags remain available for failed slices and use the same `scripts/live-smoke/.env` credentials as friendly commands.
- Failed-case rerun suggestions include provider, lifecycle mode, and policy when relevant, but do not preserve generated run ids.
- Targeting `resumed_session` includes the prerequisite `new_session` case in the same run because resume requires a fresh continuation.
- Use a full-matrix-friendly default case timeout so normal `run` commands do not require a timeout flag.
- Preserve artifacts by default under a repo-local gitignored artifact root, with override and explicit cleanup options.
- Use one run id per smoke invocation, allow path-safe user-supplied run ids, and derive per-case sentinels from the run id.
- Write a JSON summary artifact for real runs, optionally emit JSON to stdout, and use zero/non-zero process exit only.
- Serialize live-smoke JSON artifact paths with stable forward-slash separators while keeping runtime filesystem operations on native `Path` values.
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
