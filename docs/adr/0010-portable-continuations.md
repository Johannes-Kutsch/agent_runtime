# Portable continuations

Session-backed runtime execution should return an opaque portable `Continuation` resume token that callers persist and pass back to resume without relying on a runtime-managed provider state directory. A completed session-backed run must include meaningful continuation data; if a provider cannot produce resume data, the runtime should not report a successful session-backed result. Display and policy metadata such as selected service, model, effort, and `ToolPolicy` belong in result metadata rather than in the continuation contract. The runtime may carry provider-owned resume data inside the continuation, including encoded provider state when needed, but callers must not treat provider resume payloads as a stable public schema and no released consumer compatibility needs to be preserved during the refactor.

Built-in providers that cannot produce and consume portable continuation data should be limited to ephemeral execution until they can satisfy the session-backed contract. Durable invocation logging should follow the same ownership rule: the runtime may return structured invocation records, but callers own persistence and file layout.

Usage-limit grouping is also caller policy outside the core runtime API. Runtime outcomes should expose provider and service facts such as selected service, account label, reset time, and continuation state rather than a caller-defined grouping key.

Core runtime requests should not require caller-defined labels. Application correlation, workflow naming, and display grouping belong outside the runtime boundary unless a concrete runtime behavior needs a named field.

This supersedes ADR 0009's requirement that new-session and resumed-session calls require a caller-supplied `RuntimeStateDir`, that continuations carry provider state identifiers relative to that directory, and that durable invocation logging is written by the runtime under a `RuntimeLogsDir`.
