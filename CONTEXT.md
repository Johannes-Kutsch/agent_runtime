# agent_runtime Context

## Purpose

`agent_runtime` is the reusable runtime boundary for agent execution. It owns contracts that can be consumed by an application adapter without importing the application itself.

## Ubiquitous Language

| Term | Meaning |
| --- | --- |
| `agent_runtime` | The reusable runtime package and its stable core public surface. |
| `StageOverride` | A single stage selection node containing service, model, effort, and optional fallback. |
| `ServiceRegistry` | The runtime-owned resolver that maps configured services and stage chains to an executable candidate. |
| `AgentService` | The protocol implemented by provider adapters for execution behavior. |
| `RunKind` | The runtime mode for a service invocation, such as fresh or resumable. |
| `ProviderSessionState` | The provider-owned session state that records how a run should start or resume. |
| `ProviderSessionAdapter` | The narrow adapter seam that owns provider-specific session policy. |
| `WorkInvocation` | The runtime-owned work lifecycle that turns caller intent plus execution dependencies into a text result. |
| `AgentRuntimeError` | The base error for runtime failures. |

## Boundary Rules

- The runtime package must remain importable without application modules.
- Application-specific prompt rendering, CLI wiring, issue orchestration, and output parsing belong outside the runtime boundary.
- The runtime/request seam stays a single vertical flow from caller intent through session planning to work invocation.
- The package root should stay a narrow compatibility entrypoint, not a catch-all export surface.
- Runtime-owned selection, availability, and resumability policy stay in the runtime boundary.
- Provider execution behavior stays behind focused adapter contracts.
- Provider-specific session details must stay behind explicit adapter contracts.
- Runtime-owned public names should be neutral and caller-supplied where paths or log roots are involved.

## Runtime Surfaces

- One-shot prompt execution for already-rendered prompts.
- Resident execution for resumable sessions.
- Caller intent through session planning and work invocation remains one vertical flow.
- Package-root imports stay narrow while behaviorful entrypoints live under focused modules.
- Service selection across nested `StageOverride` chains.
- Provider execution behind adapter contracts.
- Provider session planning and state recovery.
- Text-output reduction from parsed provider events.
- Agent log reservation and append/update lifecycle.
