# Wiring CancellationToken into invocation execution

Status: accepted

`CancellationToken.cancel()` shipped fully unwired — `is_cancelled` was read nowhere outside its own definition. Wiring it: poll sub-second inside Built-in Provider Invocation's read loop (not gated on `timeout_seconds`), check once before subprocess spawn to skip launch entirely if already cancelled, hard-kill only (same `taskkill`/`process.kill()` path as Idle Timeout, no graceful-shutdown attempt), and cancellation wins ties against Idle Timeout. Preserve a `Continuation` only when provider work had started, mirroring Idle Timeout's existing continuation-preservation rule exactly — cancellation isn't special-cased to always discard resumability.

`.cancel()` is designed to be called from a different thread than the one blocked executing the invocation — this is the expected usage, not an edge case, since a run occupies whatever thread awaits it and can never notice its own cancellation. Consumers already isolate parallel runs onto separate threads/sandboxes rather than relying on single-event-loop `asyncio` concurrency, so the runtime's execution model stays fully synchronous; no move to `asyncio.to_thread` internally. Because cross-thread signaling is now load-bearing rather than incidental, `CancellationToken`'s internal primitive changes from `asyncio.Event` to `threading.Event` (identical public `is_cancelled`/`cancel()` shape) — `asyncio.Event` is only documented as safe within one event loop/thread, and relying on it working by accident cross-thread is the same category of gap that let this feature ship unwired in the first place.

## Considered options

- Continuation on cancel: always discard vs. always preserve vs. progress-gated (chosen, matches Idle Timeout).
- Execution model: keep synchronous (chosen) vs. wrap invocation in `asyncio.to_thread` to support same-event-loop concurrent runs — rejected because consumers already get parallelism via dedicated threads/sandboxes, so the problem `to_thread` solves doesn't exist for current usage, and it would add a thread-inside-a-thread for no benefit.
- Token primitive: `threading.Event` (chosen) vs. keep `asyncio.Event` — rejected as undocumented-safe for the cross-thread case that's now the primary use.

## Consequences

- `on_live_output` keeps firing on whatever thread the caller's blocking run call executes on, same as today — no thread-affinity change, since the execution model didn't change.
- Deterministic tests cover the continuation-preservation and precedence logic; a live-probe addition (manual, not CI) confirms the real subprocess actually dies on cancel — sliced as a separate human-in-the-loop issue from the deterministic wiring work.
