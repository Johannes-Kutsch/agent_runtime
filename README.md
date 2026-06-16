# agent_runtime

`agent_runtime` is the migration target for reusable agent runtime code that should live outside the consuming application.

## What is in this repo now

- Packaging and release scaffolding
- GitHub Actions release workflow
- Domain docs hooks for future migration work
- A smoke test so CI can validate the scaffold

## Release flow

- Pushes to `main` publish to TestPyPI.
- Tags matching `v*.*.*` publish to PyPI.

## Next step

Copy the runtime modules into `src/agent_runtime/` when you are ready to start the actual migration.

