# Live Provider Probe: manual-debug-only, supersede the CI smoke runner

Status: Accepted. Supersedes ADR 0007 (which positioned the runner for credentialed CI with stable JSON, exit-code contract, and matrix exhaustiveness).

ADR 0007 built a "Live Provider Smoke Test" that doubled as opt-in maintainer tooling *and* a credentialed CI gate: machine-readable JSON, zero/non-zero exit contract, full lifecycle × `ToolPolicy` matrix, accumulating run-id'd artifacts. In practice the only thing this tool does that the deterministic pytest suite cannot is prove a real live-API call goes through — classification (non-empty output, metadata match, continuation presence, resume sentinel echo) is already unit-tested against fakes. The CI framing carried weight (JSON, exit codes, immutable accumulation) for a use case we no longer want, and "smoke test" misnames a thing you run by hand to watch what a provider does. We reposition it.

## Decision

- **Rename to `Live Provider Probe`. Manual debugging only — CI is not a goal.** Drop the JSON summary, the exit-code contract, and the rerun-command suggestions. No machine-readable output channel.
- **Verdict = the runtime's outcome category, not a content judgement.** Per case, report `success` (`Completed`), `usage_limited`, `no_service_available`, `timed_out`, `retryable_failure`, `wrong_credentials` (`AgentCredentialFailureError`), or `error` (any other exception, with traceback). Drop the non-empty-output, metadata-match, continuation-presence, and resume-sentinel checks — pytest covers them, and re-proving them against live providers is redundant. Prompts become trivial.
- **Live terminal display.** Stream `agent_message` and `agent_tool_call` events (`display_message` only) as they arrive, under a per-case header. Anything that isn't `success` prints in red as "run not completed." Reverses ADR 0007's "does not stream subprocess output."
- **Two JSON artifacts per case.** `live_feed.json` (JSON-lines, appended as events arrive so a crash leaves a valid partial feed; carries `type`, `display_message`, and full `raw_provider_output`) and `result.json` (outcome `kind`, `selected`, `output`, `usage`, `continuation`, traceback). `raw_provider_output` is captured in full despite verbosity/sensitivity.
- **Service-keyed layout, wiped on rerun.** `artifact-root/<service>/<kind>_<ToolPolicy>/…`. Before a service reruns, its entire `<service>/` dir is deleted and recreated. Drop the run-id concept (random or user-supplied). Reruns overwrite instead of accumulate.
- **Six cases per service.** Three entry paths at `UNRESTRICTED`, plus ephemeral under each remaining `ToolPolicy`, deduplicated on `ephemeral_UNRESTRICTED`. `new_session` and `resumed_session` are a coupled pair: resume always runs new_session first and feeds its continuation in.
- **Selection.** Default = all configured providers. `all` skips unconfigured providers (no wipe, no failure); an explicitly named unconfigured provider surfaces (`wrong_credentials`/`error`, red). Single-provider form for focused runs.
- **Unchanged from 0007:** opt-in, out of default pytest and the installed Runtime Public Surface; public runtime imports only; `Live Probe Default` cost-first tuples with CLI override; credentials from `scripts/live-probe/.env` only (probe script, plan module, and `.env`/`.env.example` co-located in `scripts/live-probe/`); artifact root gitignored and marked potentially-sensitive; runs serial.

## Consequences

- One thing the probe proves that pytest can't: a real provider invocation reaches a classified outcome without an unexpected exception. Everything else a human reads off the artifacts.
- Wipe-on-rerun is hostile to CI's fresh-checkout/accumulate model — deliberate, since CI is dropped. Re-adding CI later means re-adding JSON, exit codes, and immutable artifacts.
- No machine-readable result; callers wanting one are not a supported audience.
- Live terminal stream may surface raw-ish provider chatter to the console (filtered to message/tool-call display lines); full raw stays in `live_feed.json`.
