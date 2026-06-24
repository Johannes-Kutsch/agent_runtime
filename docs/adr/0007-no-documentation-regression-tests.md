# No documentation regression tests

Documentation is project knowledge, not an executable runtime contract. The default test suite must not assert README, `CONTEXT.md`, public docs, ADR wording, or CLI help prose, because such tests make text restructuring look like runtime breakage. Tests may verify that help surfaces exist and exit correctly; runtime behavior and package artifacts remain valid test targets.
