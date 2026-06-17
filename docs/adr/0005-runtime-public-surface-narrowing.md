# Runtime public surface narrowing

The runtime boundary should expose a smaller, clearer front-facing surface. Callers should see one canonical runtime entrypoint per mode, a narrow package root, and focused seams for work invocation, session planning, and provider policy.

## Decision

- Keep the package root as a narrow compatibility entrypoint rather than a catch-all export surface.
- Expose one canonical entrypoint per runtime mode instead of parallel facade and free-function surfaces for the same behavior.
- Keep work invocation dependencies focused on runtime execution rather than presentation or orchestration concerns.
- Keep resident session planning as a value-oriented seam and keep provider-session mutation behind the provider-facing adapter.
- Keep the service registry responsible for selection and availability policy, and keep presentation helpers outside that seam.
- Preserve runtime-owned selection, resumability, and failure policy while simplifying the public shape around them.
- Treat low-level work invocation modules as runtime implementation modules, even when they remain importable for compatibility and tests.
- Replace the closed runtime-owned `AgentRole` vocabulary with caller-defined invocation labels.
- Represent invocation labels as a runtime-owned validated value object whose values are supplied by callers.
- Require canonical runtime requests to receive an explicit invocation label instead of defaulting to a runtime-owned workflow role.
- Treat usage-limit grouping as caller policy, not as an implicit mapping from invocation role.
- Record invocation labels in runtime-owned logs as `invocation_role`, not as `role`.

## Consequences

- The runtime boundary is easier to learn and test through its public surface.
- Callers have fewer equivalent ways to reach the same behavior.
- The deep parts of the runtime stay intact while the shallow surface area shrinks.
- Compatibility shims can still exist where necessary, but they no longer define the boundary shape.
- Ordinary consuming projects should integrate through the runtime entrypoints and adapter seams instead of assembling work invocation internals directly.
- Future runtime modes should carry invocation labels without expanding application workflow semantics in the runtime package.
- The runtime keeps validation and path-safety locality, while consuming projects keep ownership of their role vocabulary.
- Logs, state paths, provider commands, and usage-limit stage keys should reflect caller intent instead of an implicit `implementer` default.
- Consumers that need quota grouping separate from invocation identity should provide an explicit usage-limit scope.
- Log records should use vocabulary that matches the runtime boundary before the first release freezes the schema.
