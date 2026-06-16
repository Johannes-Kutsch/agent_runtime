# Runtime boundary closure

The runtime distribution must stay self-contained. Any module that ships in the package should import cleanly without application modules, and the built artifact should match the editable-source boundary.

## Decision

- Keep application orchestration out of the runtime distribution.
- Keep provider-specific policy behind adapter contracts.
- Require standalone importability for all shipped runtime modules.

## Consequences

- The runtime package can be tested as a standalone artifact.
- Boundary regressions show up as import failures instead of implicit coupling.
- Consumers remain responsible for application-specific orchestration and presentation.
