# Ubiquitous Language

## Package & Distribution

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **agent_runtime** | Reusable Python package boundary for shared agent runtime behavior. This repository is the migration target for runtime code that should be importable and testable without the consuming application. | runtime facade, shared utils |
| **package migration** | The boundary move that extracts reusable runtime behavior into `agent_runtime` without moving application-specific orchestration code. Completion requires package-boundary proof, not just a package rename. | code move, extraction cleanup |
| **consuming project** | A repository that depends on `agent_runtime` and adapts it for its own prompts, CLI, issue flow, or application-specific wiring. | host project, parent project |

## Repository Boundary

- This repository is intentionally small during migration setup.
- Only surrounding infrastructure lives here until the runtime code is copied in.
- Keep new concepts aligned with the package boundary above.

