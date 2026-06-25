# Windows provider host environment is an allowlist, layered once for every built-in provider

The runtime deliberately hands each provider subprocess a minimal, controlled environment rather than inheriting the host's — an isolation property that mirrors the predecessor's container-supplied base environment. On Windows this minimal env is fatal: a built-in provider command is an npm shim chain (`provider.cmd` → `cmd.exe` → `node.exe`) that needs a few system variables merely to launch. Without them the process crashes immediately with `0xC0000409` (`STATUS_STACK_BUFFER_OVERRUN`) before producing any output. The same diagnosis surfaced the argv/stdin boundary recorded in [0006](0006-provider-invocation-argv-stdin-boundary.md); this ADR covers its sibling, the environment.

We restore exactly the host system variables the launch chain needs — `PATH`, `PATHEXT`, `SystemRoot`, `ComSpec`, `WINDIR` — and nothing else. We explicitly reject merging the full host environment (which is the simpler fix and matches the predecessor's behavior of inheriting `os.environ`): a full merge silently abandons the runtime's isolation property, and once providers and consumers can come to depend on arbitrary host variables being present, clawing isolation back is hard. An allowlist keeps the leakage deliberate, auditable, and small. The five keys are proven sufficient empirically — OpenCode already runs end-to-end on Windows through the identical npm-shim chain with exactly this allowlist — so the set is grounded in observed behavior, not a guess.

The base env is layered **once** at the point where any built-in provider invocation's environment is finalized, not opted into per provider. The crash existed precisely because the allowlist was a per-provider helper that OpenCode called and Claude/Codex forgot — modeling a cross-provider concern as a per-provider one. Layering it structurally makes the footgun impossible: a future fourth provider cannot reintroduce the crash. Provider-specific environment values layer over the base.

The layer is Windows-only and a strict no-op on Mac/Linux, where the minimal env is already survivable; the fix changes no POSIX behavior.

## Consequences

- Host environment leakage on Windows is bounded to five named system variables; no provider can widen it implicitly.
- Adding a built-in provider requires no Windows-env wiring — it is inherited structurally.
- The allowlist is intentionally minimal: a provider that needs more host context (e.g. Codex resolving its auth home from `USERPROFILE` on an ephemeral run) is a distinct, separately-tracked concern, not a reason to broaden the shared base.
- Mac/Linux provider isolation is unchanged; tests pin the anti-leak property on both platforms.
