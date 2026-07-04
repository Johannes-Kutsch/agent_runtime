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

## Amendment: Continuation workspace paths are serialized in POSIX format

The `Continuation` token serializes the `ToolAccess` workspace path using `Path.as_posix()` rather than `str(Path)`. On Windows, `str(Path)` produces backslash-separated strings that vary by host; `as_posix()` produces forward-slash strings that are schema-stable across platforms. Deserialization uses `Path(posix_string)`, which correctly reconstructs an absolute Windows path (e.g. `Path("C:/Users/x")` → `WindowsPath('C:/Users/x')`), so round-trip fidelity is preserved on both platforms.

This does not make the `Continuation` portable across operating systems — the Session Store is a host-native directory and cross-OS resume is not a supported use case. It makes the *token schema* host-independent: the serialized JSON contains the same path string regardless of which platform emitted the token.

## Consequences

- One uniform resume model across providers; no provider-specific embedding/rehydration code path to keep correct.
- Resumability is honest: a `Continuation` from a session-backed run is resumable when both the token and its Session Store are present *and* the provider-side session state is intact. If the provider-side session state has been lost (e.g. the provider discarded the session independently of the token), Resume Session Run raises `ContinuationUnrecoverableError` with `service_name` identifying the provider. The consumer owns the decision to drop the continuation and re-plan.
- Consumers own durable session storage and its retention/cleanup; the runtime owns none. This supersedes ADR 0004's "all resume state round-trips through the Continuation" — resume now requires the Continuation **and** the caller-owned Store.
- Agent runs never touch the developer's personal provider history; ephemeral isolation requires seeding auth into the scratch home.

## Amendment: Lost provider-side session state raises ContinuationUnrecoverableError

The Session Store model keeps the provider's native state as the single source of truth, but the provider can lose a session independently of the local continuation token — for example, Codex discards its session state when interrupted mid-run. When Session-backed Provider State Resolution detects that rollout paths exist but no session id can be recovered, the continuation cannot be honored. This is not a configuration error (the caller did nothing wrong) and not a provider unavailability (the provider itself is reachable).

The runtime raises `ContinuationUnrecoverableError(service_name=...)` — a direct sibling of `AgentRuntimeError`, distinct from `RuntimeConfigurationError` — when a Resume Session Run or Start Session Run detects an unrecoverable continuation. It carries `service_name` for diagnostics. The consumer catches it, drops the stale continuation, and re-plans; the runtime takes no automatic action.

## Amendment: The `already in use` signal was self-inflicted; the real second trigger is `No conversation found`

An earlier draft of this amendment treated two Claude CLI strings as evidence that the provider's server-side session was gone, and detected both in Built-in Provider Stream Interpretation:

- `Session ID <uuid> is already in use.`
- `No conversation found with session ID: <uuid>`

Investigating pycastle#1946 (and the downstream pycastle#1954) more closely retracts the diagnosis of the first string. `already in use` was **not** a stale server-side lock lingering for ~19 hours — a provider-side lock would not survive that long. It was `ar` **self-inflicting the collision on every scheduled retry**: `resolve_claude_resumed_session_facts` silently downgraded a resume to a fresh start when the probe dir looked empty (`RunKind.FRESH`) while **keeping the continuation's original session id**, so rendering emitted `claude --session-id <original-id>`. Because the probe dir stayed empty across retries, `ar` deterministically re-rendered the identical fresh-start-with-a-taken-id invocation every time, and Claude answered `already in use` every time. The string was a symptom of an `ar` bug (routing a *fresh* start through a *used* id), not a report that the session was dead.

This splits into a structural fix and a narrowed detection:

**Structural (the actual fix).** A resume with absent local session state must **raise `ContinuationUnrecoverableError` and hand back to the consumer** — never downgrade to a fresh start, and never reuse the continuation's id for a fresh start. Codex already did this (the first amendment's rollout-path guard). Claude and OpenCode are brought into line: Claude stops downgrading-and-reusing, and OpenCode gains the pre-flight `_opencode_is_resumable` guard it lacked on the resume path (it previously forced `--session <id>`, minting a brand-new id to "resume" a session that never existed rather than reporting the continuation unrecoverable). All three now behave identically: absent local state ⇒ raise, consumer re-plans, and a genuinely new session always gets a freshly-minted, never-before-used id. With no path that starts a session under a caller-influenced id, `already in use` is structurally impossible from a correct runtime; if it ever surfaces it is a plain hard error, not `ContinuationUnrecoverableError`.

**Detection (the narrowed second trigger).** One case still escapes the pre-flight check: the local Session Store and continuation token are valid *and* the pre-flight probe passes, yet the id is absent in the provider home Claude actually read — e.g. a consumer overrides `CLAUDE_CONFIG_DIR` so `ar`'s probe dir and Claude's real home diverge. The provider only reveals this by being invoked and replying `No conversation found with session ID: <uuid>`. Built-in Provider Stream Interpretation's Claude-specific reducer recognizes that single signal (loose substring match on the CLI's known phrasing, read from the JSON `result` event's `errors` field) and raises `ContinuationUnrecoverableError` instead of letting it fall through as a generic `HardAgentError` or, as before, a misclassified `TransientAgentError`. The `already in use` bare-stderr recognition path is dropped — that string is now prevented, not detected.

We deliberately did not generalize this to "any Resume Session Run failure is unrecoverable." `Resumed-session failures do not invalidate continuations or trigger automatic Consumer Fallback inside runtime` remains the default — a transient hiccup (rate limit, network blip, timeout) must not discard a perfectly resumable continuation. Only the `No conversation found` signal raises `ContinuationUnrecoverableError` from stream interpretation; everything else keeps its existing classification.

`ContinuationUnrecoverableError` also carries optional `classification` (which known signal fired) and `raw_message` (the exact provider text), mirroring the shape already used by `HardAgentError.classification` and `ModelNotAvailableError.raw_message`, so a consumer catching it has structured access to both without parsing `str(exc)`.

The stream-detection trigger is Claude-only: OpenCode has no observed analogous CLI signal, and Codex leans on its pre-flight guard. The pre-flight raise-on-absent-state guard, by contrast, is now uniform across all three session-backed providers.
