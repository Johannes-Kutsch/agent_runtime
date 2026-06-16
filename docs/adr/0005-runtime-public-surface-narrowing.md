# Runtime public surface narrowing

The runtime boundary should expose a smaller, clearer front-facing surface. Callers should see one canonical runtime entrypoint per mode, a narrow package root, and focused seams for work invocation, session planning, and provider policy.

## Decision

- Keep the package root as a narrow compatibility entrypoint rather than a catch-all export surface.
- Expose one canonical entrypoint per runtime mode instead of parallel facade and free-function surfaces for the same behavior.
- Keep work invocation dependencies focused on runtime execution rather than presentation or orchestration concerns.
- Keep resident session planning as a value-oriented seam and keep provider-session mutation behind the provider-facing adapter.
- Keep the service registry responsible for selection and availability policy, and keep presentation helpers outside that seam.
- Preserve runtime-owned selection, resumability, and failure policy while simplifying the public shape around them.

## Consequences

- The runtime boundary is easier to learn and test through its public surface.
- Callers have fewer equivalent ways to reach the same behavior.
- The deep parts of the runtime stay intact while the shallow surface area shrinks.
- Compatibility shims can still exist where necessary, but they no longer define the boundary shape.
