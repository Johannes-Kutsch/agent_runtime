# Resume state lives in a caller-owned Session Store; the runtime never recreates it

Session-backed providers (Claude, Codex, OpenCode) keep their conversation state in a native on-disk home directory and resume by **session id**: the runtime invokes `--resume <id>` / `resume <id>` / `--session <id>` plus the new prompt, and the provider CLI reloads the prior transcript from its home. The runtime never re-feeds the transcript to the model — the id is only a lookup key, and the transcript must exist on disk in the provider's home at resume time.

The runtime deliberately redirects each provider's home (`CLAUDE_CONFIG_DIR` / `CODEX_HOME` / `OPENCODE_HOME`) away from the host user's native directory to an isolated location, so agent runs neither read nor pollute the developer's personal provider history. That isolation is the whole reason resume needs an explicit durable location: there is no shared global storage to fall back on.

Two approaches existed in the codebase. Claude and Codex keep the bytes on disk and their `Continuation` carries only resume identity (Codex additionally a relative path pointer; Claude, by an unnoticed bug, nothing). OpenCode instead **embedded** its on-disk state (`resume.jsonl` + session id) inside the portable `Continuation` and rehydrated a throwaway temp directory before resuming. The probe exposed the gap: when no durable location was supplied, new-session wrote to a temp directory that was deleted in a `finally`, so Claude/Codex resume failed (`No conversation found with session ID`), while OpenCode's embedding masked it.

## Decision

- Resume uses a **Session Store**: a caller-owned, isolated directory holding the provider's native on-disk session state. It is a **public, required** input for `Start Session Run` and `Resume Session Run`, supplied **symmetrically** to both. Omitting it on a session-backed run is a configuration error. (`Ephemeral Run` needs no Store; it gets a throwaway isolated scratch home.)
- The runtime **never serializes, embeds, or recreates** provider session state. It points the provider home at the Store, preserves the Store across calls, and lets the provider resolve its own session by id. We **drop OpenCode's embedded-state path** so all three providers behave identically.
- The `Continuation` stays an opaque portable token carrying resume identity plus a **relative-path pointer** into the Store. Every Session-backed Provider's continuation carries that pointer — fixing Claude, which carried none.
- Every run executes against an **isolated provider home**, never the host user's native home; credentials are seeded into the isolated home from the host (including ephemeral Codex, which previously leaned on `~/.codex`).

## Rationale

We reject the self-sufficient-continuation (embedding) model as the universal approach despite OpenCode proving it works. Embedding requires the runtime to know and faithfully serialize each provider's on-disk layout; Claude's resumable state is an opaque directory tree the runtime does not model, so embedding it is fragile by construction. Worse, a snapshot taken when a continuation is emitted silently loses any tool-call/session work the provider records outside the one file we capture, and loses work entirely if a resumed run is interrupted before emitting a fresh continuation (its temp home is cleaned up). The Session Store keeps the provider's native state as the single source of truth, untouched and provider-agnostic — lossless and crash-resilient. The cost we accept: the `Continuation` is no longer resumable on its own — it must travel with its Session Store — trading single-token portability for not losing work.

## Consequences

- One uniform resume model across providers; no provider-specific embedding/rehydration code path to keep correct.
- Resumability is honest: a `Continuation` from a session-backed run is always resumable, because the run required a durable Store.
- Consumers own durable session storage and its retention/cleanup; the runtime owns none. This supersedes ADR 0004's "all resume state round-trips through the Continuation" — resume now requires the Continuation **and** the caller-owned Store.
- Agent runs never touch the developer's personal provider history; ephemeral isolation requires seeding auth into the scratch home.
