# Provider invocation argv and stdin boundary

Built-in provider invocation must be cross-platform runtime behavior, not host shell script rendering. The internal Built-in Provider Invocation Seam models provider execution as an executable plus arguments plus explicit prompt input, avoiding shell redirection strings as the canonical representation because quoting rules differ across host shells and are easy to misapply.
