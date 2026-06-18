# Session-backed result vocabulary

The lifecycle runtime API uses **Start Session Run** and **Resume Session Run** instead of the older one-shot/resumable split, so shared completed values should not keep the legacy resumable runtime-mode vocabulary. Use `SessionRunResult` and `SessionRuntimeMetadata` as the canonical public names for completed session-backed execution results and metadata, replacing `ResumableRunResult` and `ResumableRuntimeMetadata` before the first release.
