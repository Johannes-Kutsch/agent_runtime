# Silent provider hangs stay honest timeouts; Idle Timeout owns subprocess termination

A provider subprocess (first seen with expired/exhausted OpenCode Go subscriptions) can hang silently forever — no output, no non-zero exit. The cause is ambiguous from the runtime's side: exhausted quota and a server-side maintenance outage are indistinguishable. We classify the resulting kill as a plain `timed_out` outcome and explicitly reject fabricating a `UsageLimited` outcome with a synthetic reset to steer consumer fallback — that would lie in the maintenance case and would make the runtime decide Consumer Fallback eligibility, which is the consumer's job. Consumers already get `ResolvedProvider` on the outcome and can implement back-off themselves.

To make a terminating timeout possible at all, the Idle Timeout moves into Built-in Provider Invocation (the only layer holding the process handle), reads provider output against a deadline, resets on any raw output line, and kills the process on silence. The prior event-layer watchdog — which could only raise on the *next* Agent Event and so never fired on total silence — is retired. This reverses the earlier stance that no time/threading concern belonged near the output context.

## Consequences

- OpenCode timeout cause stays ambiguous by design; the quirk is documented for consumers rather than papered over.
- Idle Timeout resets on raw output lines, not interpreted Agent Events — slightly more lenient liveness.
- Pre-flight subscription checks rejected: no OpenCode usage endpoint exists, and invalid keys are already rejected up front.
