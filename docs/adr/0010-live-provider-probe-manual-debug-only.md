# Live Provider Probe: manual-debug-only

Repositioned from credentialed-CI smoke runner to manual debugging tool. The only thing the probe proves that pytest can't: a real provider invocation reaches a classified outcome without an unexpected exception.

## Decision

- **Manual debugging only — CI is not a goal.** No JSON summary, exit-code contract, or rerun suggestions.
- **Verdict = outcome category, not content judgement.** Per case: `success` (`Completed`), `usage_limited`, `provider_unavailable`, `timed_out`, `wrong_credentials`, or `error` (with traceback). No non-empty-output, metadata-match, or continuation-presence checks — pytest covers those.
- **Live terminal display.** Stream `agent_message` and `agent_tool_call` `display_message` as they arrive. Non-success prints red as "run not completed."
- **Two JSON artifacts per case.** `live_feed.json` (JSON-lines with `type`, `display_message`, `raw_provider_output`) and `result.json` (outcome `kind`, `selected`, `output`, `usage`, `continuation`, traceback).
- **Service-keyed layout, wiped on rerun.** `artifact-root/<service>/<kind>_<ToolPolicy>/…`. Reruns overwrite, not accumulate.
- **Six cases per service.** Three entry paths at `UNRESTRICTED`, plus ephemeral under each remaining `ToolPolicy`, deduplicated on `ephemeral_UNRESTRICTED`. Resume always runs new_session first.
- **Selection.** Default = all configured; skips unconfigured. Explicit unconfigured provider surfaces error. Single-provider form for focused runs.
- Opt-in, out of default pytest and Runtime Public Surface; public runtime imports only; `Live Probe Default` cost-first tuples with CLI override; credentials from `scripts/live-probe/.env` only; artifact root gitignored; runs serial.

## Consequences

- Wipe-on-rerun hostile to CI — deliberate. Re-adding CI means re-adding JSON, exit codes, immutable artifacts.
- No machine-readable result; callers wanting one are not supported.
- Live terminal stream may surface raw-ish provider chatter; full raw stays in `live_feed.json`.
