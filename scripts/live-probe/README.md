# Live Provider Probe

Manual-debug-only tool that invokes the real built-in providers through the
Runtime Public Surface so you can watch what they actually do. It is **not** CI,
not part of the default test suite, and not a Runtime Public Surface addition.
The one thing it proves that the pytest suite can't is that a real provider
invocation reaches a classified runtime outcome without an unexpected
exception. See [ADR 0013](../../docs/adr/0013-live-provider-probe-manual-debug-only.md).

## Setup

Copy the template and fill in credentials (read-only; never written back):

```
cp .env.example .env
```

`.env` is gitignored. Codex uses host auth (`~/.codex/auth.json`) instead of a
key in `.env`.

## Run

```
# probe all configured providers
python scripts/live-probe/live_provider_probe.py

# focus one provider
python scripts/live-probe/live_provider_probe.py claude

# override model / effort (defaults are cost-first)
python scripts/live-probe/live_provider_probe.py claude --model claude=sonnet --effort claude=high
```

Unconfigured providers are skipped silently under the all-providers run, but a
provider you name explicitly is surfaced in red.

## Output

Per provider, six cases run (the three entry paths at `UNRESTRICTED`, plus
ephemeral under each other `ToolPolicy`). Agent messages and tool calls stream
live to the terminal; anything that isn't `success` prints red as
"run not completed".

Artifacts land under `live-probe-artifacts/<service>/<mode>_<ToolPolicy>/`
(gitignored, **wiped per service on each rerun**, may contain sensitive
provider output):

- `live_feed.json` — JSON-lines, appended as events arrive (a crash leaves a
  valid partial feed); carries the full `raw_provider_output`.
- `result.json` — outcome category/kind, selected provider, output, usage,
  continuation, and any traceback.

There is no machine-readable summary, no exit-code contract, and no rerun
suggestions by design — read the terminal and the per-case `result.json`.

## Cancel-mid-turn probe

A separate entry point exercises real cancellation against a live provider
subprocess. It confirms the behaviour wired by issues #436 (ephemeral cancel →
`Cancelled` outcome, no continuation) and #437 (session-backed cancel after
provider output → `Cancelled` outcome **with** a continuation):

```
# cancel-mid-turn probe for all configured providers
python scripts/live-probe/live_provider_probe_cancel.py

# focus one provider
python scripts/live-probe/live_provider_probe_cancel.py claude

# override model / effort
python scripts/live-probe/live_provider_probe_cancel.py claude --model claude=sonnet --effort claude=high
```

Per provider, two cases run. Each starts a real invocation, waits for the
first live output event (so the provider subprocess is running), then calls
`.cancel()` on the `CancellationToken`:

- `cancel_ephemeral_UNRESTRICTED` — ephemeral invocation; expects outcome
  `Cancelled` with no continuation. Terminal reports "subprocess terminated"
  (confirmed by the `Cancelled` outcome: the runtime hard-kills the process).
- `cancel_new_session_UNRESTRICTED` — session-backed invocation; expects
  outcome `Cancelled` **with** a continuation (provider work had started).
  Terminal reports "continuation returned: yes/no".

Artifacts land under the same `live-probe-artifacts/<service>/` root (wiped
per service on each rerun) following the same layout as the standard probe:

- `live_feed.json` — JSON-lines of live events up to the cancellation point.
- `result.json` — outcome category/kind, continuation (if any), and traceback.
