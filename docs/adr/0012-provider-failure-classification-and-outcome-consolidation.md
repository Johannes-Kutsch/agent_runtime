# Provider failure classification and outcome consolidation

`RetryableProviderFailure` and `NoServiceAvailable` consolidate into `ProviderUnavailable` with closed reason enum (`SERVICE_NOT_AVAILABLE`, `TRANSIENT_API_ERROR`). Internal exceptions merge into `ProviderUnavailableError`. `ProviderErrorObservation` dropped in favor of raw error messages.

Driven by discovering that a provider subprocess failure (non-zero exit, missing binary) was silently classified as `Completed` because the adapter never checked exit codes. The failure taxonomy conflated expected temporary failures (outcome values) with hard/credential failures (exceptions).

## Consequences

- Expected provider failures (temporary unavailability, transient API errors) are return values, not exceptions. Hard and credential failures remain exceptions.
- `ProviderErrorObservation` deleted; diagnostics use raw error messages.
- Process-level failures (non-zero exit, empty output) surface as `HardAgentError`.
- `reset_time` dropped from `ProviderUnavailable`; only meaningful for `UsageLimited`.
