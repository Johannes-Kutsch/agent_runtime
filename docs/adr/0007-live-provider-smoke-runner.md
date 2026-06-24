# Live provider smoke runner

Status: Accepted; artifact set partially superseded by ADR 0012 (invocation records removed from the runtime surface, so they are no longer a preservable artifact).

Deterministic tests cover lifecycle behavior, command rendering, tool policy, live output observation, and portable continuations. They do not prove real Claude, Codex, and OpenCode integrations can invoke provider commands with host credentials and live availability. The project needs an opt-in Live Provider Smoke Test for maintainers and credentialed CI that proves the Runtime Public Surface contract without becoming a provider benchmark, default suite dependency, or runtime public API.

## Decision

- Opt-in standalone live smoke runner as maintainer tooling under `scripts/`; live provider execution stays out of default pytest and the installed Runtime Public Surface.
- Default pytest may cover smoke-runner planning/artifact behavior only when provider auth, availability, paths, and invocation are injected or faked.
- Use only public runtime imports/request values (`RuntimeClient`, lifecycle requests, `ProviderSelection`, `ProviderAuth`, `Continuation`, public `ToolPolicy`); internal vocabulary (`RuntimeStateDir`, `RuntimeLogsDir`, `SessionNamespace`, `ToolAccess`, `ToolPolicyProfile`) is not smoke-runner surface.
- Provider selection is explicit: `claude`, `codex`, `opencode`, multiple explicit providers, or `all`. Explicit selection treats missing config as a configuration error; `all` skips unconfigured providers but fails when none are configured. Friendly per-provider commands keep explicit semantics; friendly all-configured commands keep `all` semantics.
- Resolve model/effort with precedence `CLI override > hardcoded smoke default`; no shell environment variables for configuration. Defaults are chosen from the cheapest runtime-supported provider/model/effort tuples (lowest supported effort), overridable by maintainers; defaults only fill missing model/effort and leave credential semantics unchanged. Smoke prompts stay simple enough for those defaults.
- Load credentials only from fixed `scripts/live-smoke/.env` (`scripts/live-smoke/.env.example` committed template); never from shell env, `RuntimeClient`, or public lifecycle entrypoints. Tooling reads but never writes/mutates credential files. Parse with a small grammar: non-empty lines are simple `KEY=value`, whitespace trimmed, malformed lines fail fast with line numbers. Absent `.env` is valid for all-configured runs when only host-auth providers (e.g. Codex) are configured; missing/blank/whitespace explicit-credential values are missing credentials.
- Support lifecycle, tool-policy, and combined smoke modes. Lifecycle smoke proves invocation health and session continuity (non-empty outputs, meaningful continuation, resumed output containing the prior sentinel) without requiring exact provider output. Capture Live Runtime Output when available; absence of live turns must not fail the run. Tool-policy smoke uses Ephemeral Run only and proves successful invocation under each public `ToolPolicy`, not tool usefulness, sandbox enforcement, or mutation behavior.
- Do not test consumer fallback orchestration, cancellation, every model/effort combination, custom tool-policy profiles, or raw provider streams.
- Run provider cases serially; in combined mode group by provider, run lifecycle first, and skip a provider's tool-policy matrix if lifecycle failed.
- No dry-run or provider-listing modes; commands execute providers directly (negligible cost) and report missing configuration at run start.
- Friendly full-matrix subcommands: `run` for all configured providers, `run <provider>` for one explicit provider; lower-level flags for targeted debugging. A provider selection without targeting flags means the Full Live Smoke Matrix; lifecycle/policy flags narrow it. Targeting `resumed_session` includes the prerequisite `new_session` case (resume needs a fresh continuation). Failed-case rerun suggestions include provider, lifecycle mode, and policy, but not generated run ids.
- Preserve artifacts by default under a repo-local gitignored artifact root, with override and explicit cleanup options. One run id per invocation; allow path-safe user-supplied run ids; derive per-case sentinels from the run id. Write a JSON summary artifact, optionally emit JSON to stdout, use zero/non-zero exit only, and serialize artifact paths with stable forward-slash separators while runtime filesystem ops stay on native `Path`.
- Preserve runner diagnostics (config summaries, prompts, final outputs, live turns, outcome summaries, invocation records when returned, timings, tracebacks). Never preserve credentials, environment dumps, auth files, credential-derived values, or opaque home-directory provider state. Do not auto-redact provider output; mark artifacts as potentially sensitive and keep them under the ignored artifact root.

## Consequences

- Maintainers verify real built-in provider invocation without making normal tests depend on live services, credentials, quota, or network.
- The runner exercises the same public lifecycle entrypoints ordinary consumers use.
- Results identify invocation, continuation, configuration, availability, and artifact failures, but do not certify answer quality or tool behavior.
- CI can opt into protected credentialed smoke jobs with stable JSON output.
- Runner-created artifacts may be sensitive; the artifact root stays ignored and credential capture is avoided.
