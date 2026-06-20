# No documentation regression tests

Documentation is project knowledge, not an executable runtime contract. The default test suite should not assert README, `CONTEXT.md`, public docs, ADR wording, or CLI help prose because these tests make text restructuring look like runtime breakage; tests may verify that help surfaces exist and exit correctly, while runtime behavior and package artifacts remain valid test targets.
