# Triage Labels

The skills speak in terms of canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

*Which field* carries each role — a label, a native status, a body line — lives in `docs/agents/issue-tracker.md` under **Mechanisms**. This file only names the strings, and applies to whichever axes that file routes to labels.

## State roles

| Canonical role    | Label in our tracker | Meaning                                  |
| ----------------- | -------------------- | ---------------------------------------- |
| `needs-triage`    | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`      | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent` | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human` | `ready-for-human`    | Requires human implementation            |
| `wontfix`         | `wontfix`            | Will not be actioned                     |

## Category roles

| Canonical role | Label in our tracker | Meaning                    |
| -------------- | -------------------- | -------------------------- |
| `bug`          | `bug`                | Something is broken        |
| `enhancement`  | `enhancement`        | New feature or improvement |

## Slice-mode labels

Used by `to-tickets` and `triage` to mark how an AFK ticket should be implemented. See `_shared/SLICE-MODES.md` for the definitions and the one-mode-per-ticket invariant.

| Canonical label    | Label in our tracker | Meaning                                                                   |
| ------------------ | -------------------- | ------------------------------------------------------------------------- |
| `behavior-slice`   | `behavior-slice`     | Introduces or changes observable behavior verifiable by a test            |
| `refactor-slice`   | `refactor-slice`     | Changes structure without changing observable behavior                    |
| `docs-slice`       | `docs-slice`         | Markdown-only change (CONTEXT.md, ADRs, README)                           |
| `needs-slice-type` | `needs-slice-type`   | Flags a malformed `ready-for-agent` issue (missing/multiple mode labels)  |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from these tables.

Edit the right-hand columns to match whatever vocabulary you actually use.
