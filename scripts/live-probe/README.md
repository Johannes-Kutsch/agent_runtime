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
