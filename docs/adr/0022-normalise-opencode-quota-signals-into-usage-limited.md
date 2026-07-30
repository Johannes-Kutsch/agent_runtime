# Normalise OpenCode quota-exhaustion signals into UsageLimited

Status: active

OpenCode surfaces quota exhaustion through two channels that ar was not normalising: an idle timeout (silent hang, no structured error event) and a credential failure (`401 invalid api key`, `AuthenticationError`). Both arrived at consumers as non-outcome exceptions — `AgentTimeoutError` via `TimedOut` and `AgentCredentialFailureError` respectively — forcing the consuming project to carry OpenCode-specific detection logic and branch on `service_name`.

## Two gaps

**Idle timeout.** When OpenCode exhausts its quota it emits no error event; it simply stops responding. ar's idle watchdog fires and raises `ProviderInvocationTimedOutError`, which `Runtime Outcome Folding` folds into `TimedOut`. No retry or re-interpretation is possible at that point because the only signal is silence. Every OpenCode timeout is therefore a quota signal, not a transient liveness failure.

**401 credential failure.** OpenCode uses `401 / AuthenticationError / "invalid api key"` to report permanent account exhaustion — not system misconfiguration. `Built-in Provider Parsed Output` already identifies this precisely (`classification="operator_actionable_agent_credential_failure"`). `Provider Output Reduction` then raised it as `AgentCredentialFailureError`, the same type used for genuine credential misconfigurations on other services.

## Considered options

### Idle timeout

- **New `ProviderUnavailableReason` variant.** Rejected: temporary unavailability is the wrong category; the account is quota-exhausted, not the service.
- **Provider-specific timeout hook in `Built-in Provider Invocation`.** Rejected: the invocation layer owns subprocess mechanics, not outcome semantics.
- **Reclassify in `Runtime Outcome Folding` based on `selected_provider.service`.** Accepted. Outcome folding already holds both the exception and the resolved provider; checking `service == "opencode"` is a one-liner at the decision point where the outcome type is chosen.

### Credential failure

- **New `ProviderUnavailableReason.PERMANENT_CREDENTIAL_FAILURE`.** Rejected: the signal is a quota/exhaustion signal, not a provider-availability signal. Adding a credential-flavoured reason to `ProviderUnavailable` would leak the implementation detail that OpenCode overloads 401 for account exhaustion.
- **Keep `AgentCredentialFailureError`, add a flag for consumers to read.** Rejected: consumers would still need OpenCode-specific knowledge to act on the flag. The type would carry a dual meaning.
- **Reclassify as `UsageLimitError(is_permanent=True)` in `Provider Output Reduction`.** Accepted. `UsageLimit` in `contracts.py` already carries `is_permanent: bool`; the parallel field on `UsageLimitError` exists. The `classification="operator_actionable_agent_credential_failure"` marker in `Built-in Provider Parsed Output` already precisely identifies this case. `Provider Output Reduction` checks the classification and raises `UsageLimitError(is_permanent=True)` instead of `AgentCredentialFailureError`.

## Consequences

- `UsageLimited` (the `RuntimeOutcome` kind) gains `is_permanent: bool = False`. `Runtime Outcome Folding` propagates `exc.is_permanent` into it. Additive, backwards-compatible.
- Two new `UsageLimited` outcome paths for OpenCode:
  - Idle timeout: `Runtime Outcome Folding` returns `UsageLimited(reset_time=None, is_permanent=False)` when `AgentTimeoutError` is caught and `selected_provider.service == "opencode"`.
  - 401: `Provider Output Reduction` raises `UsageLimitError(is_permanent=True)`, which `Runtime Outcome Folding` folds into `UsageLimited(reset_time=None, is_permanent=True)`.
- `AgentCredentialFailureError` is no longer raised for OpenCode's `operator_actionable_agent_credential_failure` case. Consumers that previously caught `AgentCredentialFailureError` for OpenCode must now handle `UsageLimited(is_permanent=True)` in the normal-return path. This is a breaking change on that exception path.
- Accounts with multiple OpenCode credentials rotate at the consumer level; ar performs no in-runtime account rotation. This is consistent with ar's existing policy: runtime classifies, consumers act.
- All other credential failures (Claude subscription denial, Codex auth lineage exhausted, any unrecognised OpenCode credential signal) remain `AgentCredentialFailureError`. The classification gate introduced here is narrow: exactly `classification == "operator_actionable_agent_credential_failure"`.
