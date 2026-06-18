# Resumed-session request construction

`ResumedSessionRunRequest` is the canonical public request name for resumed-session execution. It should keep both construction paths that the public API already describes: ordinary consumers resume from a `Continuation`, while advanced consumers may provide a lower-level session plan directly; the older `ResumableRunRequest` spelling should be removed before release rather than preserved as a compatibility alias.
