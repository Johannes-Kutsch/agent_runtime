# Provider failure classification and outcome consolidation

`RetryableProviderFailure` and `NoServiceAvailable` outcome kinds are consolidated into a single `ProviderUnavailable` kind with a closed reason enum (`SERVICE_NOT_AVAILABLE`, `TRANSIENT_API_ERROR`). The corresponding internal exceptions (`RetryableProviderFailureError`, `NoServiceAvailableError`) merge into `ProviderUnavailableError`. Structured `ProviderErrorObservation` is dropped at all seams in favor of raw error messages.

This was driven by discovering that a provider subprocess failure (non-zero exit, missing binary) was silently classified as `Completed` because the adapter never checked exit codes. Fixing the bug surfaced that the failure taxonomy conflated two concerns: expected temporary failures (outcome values) vs hard/credential failures (exceptions). The rename and consolidation align the model with that split.

## Considered options

**Keep `RetryableProviderFailure` and `NoServiceAvailable` as separate outcome kinds.** Rejected because from a consumer's perspective both mean "provider couldn't do the work, try later" — the pre-invocation vs during-invocation distinction is an internal detail, not a consumer branching point.

**Add a generic `ProviderFailure` outcome kind with an open `classification: str` field.** Rejected because the existing outcome model uses specific kinds (`UsageLimited`, `TimedOut`, `Cancelled`), not generic buckets with discriminator strings. An open string would also hide from consumers what failure reasons to expect.

**Keep `ProviderErrorObservation` for structured diagnostics.** Rejected because observations were never read after construction, the structured fields were populated inconsistently, and splitting raw messages into fixed fields risks losing context.

## Consequences

- Expected provider failures (temporary unavailability, transient API errors) travel as return values through the invocation adapter, not exceptions. Hard and credential failures remain exceptions.
- `ProviderErrorObservation` is deleted; all provider failure diagnostics use raw error messages.
- Process-level failures (non-zero exit, empty output) surface as `HardAgentError`, not as outcome values.
- `reset_time` is dropped from `ProviderUnavailable`; it was only meaningful for `UsageLimited`.
