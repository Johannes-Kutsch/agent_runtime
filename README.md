# agent_runtime

`agent_runtime` is a reusable Python runtime package for agent execution contracts, session planning, service selection, and log lifecycle management.

Install the distribution as `ruhken-agent-runtime` and import it as `agent_runtime`.

```bash
pip install ruhken-agent-runtime
```

## Layout

- `src/agent_runtime/` - runtime implementation
- `tests/` - package contract tests
- `context.md` - concise domain context for agents
- `docs/adr/` - architecture decisions that define the runtime boundary

Execution entrypoints live under `agent_runtime.runtime`; the package root stays narrow and exposes only the stable shared vocabulary.

## Consumer Integration

Ordinary consumers should start with the canonical runtime entrypoints under `agent_runtime.runtime` together with the small package-root vocabulary such as `InvocationRole`, `StageSelection`, and `ExecutionProvider`.

### One-shot Execution

One-shot execution is the normal path for an already-rendered prompt. The caller provides an explicit `InvocationRole` so logs, provider metadata, and state partitioning reflect caller intent.

```python
from agent_runtime import ExecutionProvider, InvocationRole, StageSelection
from agent_runtime.runtime import one_shot

provider: ExecutionProvider = build_provider()
selection = StageSelection(service="openai/default", model="gpt-5", effort="medium")

result = await one_shot(
    prompt=rendered_prompt,
    invocation_role=InvocationRole("issue-triage"),
    stage_selection=selection,
    provider=provider,
)

print(result.output_text)
print(result.selected_service)
```

Use the one-shot entrypoint when prompt rendering, issue orchestration, and application policy already live in the consuming application and the runtime only needs to execute the prepared prompt.

### Advanced Topics

Resumable execution, provider session planning, and `ProviderSessionAdapter` integration are advanced seams. Use them when you need provider-backed continuity, recovery, or session-specific policy, but start with the one-shot path first.

Tool-capable execution is a separate boundary from one-shot prompt execution. If an entrypoint can grant tool access, the caller must provide an explicit tool policy rather than relying on an implicit default. Follow the focused tool-policy documentation for that integration so provider-specific command rendering stays behind adapter contracts.

`ToolPolicyProfile` remains provider-neutral runtime data. Provider adapters translate runtime-owned tool policy values into backend-specific flags, arguments, and restrictions.

## Surface Boundaries

Use these as the normal integration path:

- `agent_runtime.runtime` canonical entrypoints for one-shot and resumable execution
- Package-root vocabulary such as `InvocationRole`, `StageSelection`, and `ExecutionProvider`

Use these only when you are building or extending adapters:

- Focused session-planning and provider-session seams
- Provider-policy and provider-specific DTO modules
- Provider-session path or recovery helpers

Do not treat low-level work invocation modules as the ordinary consumer API. Direct `invoke_work` usage is an advanced implementation seam, not the normal way to integrate a runtime consumer.

## Development

Install the project in editable mode with dev dependencies, then run the test suite.

```bash
pip install -e ".[dev]"
pytest
```
