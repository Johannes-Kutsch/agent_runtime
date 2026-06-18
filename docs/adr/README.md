# Architecture Decision Records

- [0001 - Runtime ownership migration](0001-runtime-ownership-migration.md): Establish `agent_runtime` as the reusable package boundary for execution contracts, session planning, service selection, and log lifecycle behavior.
- [0002 - Runtime compatibility artifact policy](0002-runtime-compatibility-artifact-policy.md): Keep public names neutral and treat any compatibility-specific artifacts as adapter concerns rather than core runtime concepts.
- [0003 - Stage resolution and failure policy](0003-stage-resolution-and-failure-policy.md): Resolve nested stage chains in priority order and classify provider failures using runtime-owned error categories.
- [0004 - Runtime boundary closure](0004-runtime-boundary-closure.md): Keep application orchestration out of the runtime distribution and require every shipped module to remain standalone-importable.
- [0005 - Runtime public surface narrowing](0005-runtime-public-surface-narrowing.md): Narrow the package root, collapse duplicate runtime entrypoints, and keep work and session seams focused on runtime intent.
- [0006 - Runtime session lifecycle entrypoints](0006-runtime-session-lifecycle-entrypoints.md): Model runtime execution around ephemeral, new-session, and resumed-session lifecycle entrypoints with consumer-owned continuation state.
