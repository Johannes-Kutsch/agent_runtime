from pathlib import Path


def test_package_imports() -> None:
    import agent_runtime

    assert agent_runtime.__name__ == "agent_runtime"


def test_readme_guides_consumers_to_ephemeral_runtime_entrypoint() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "## Consumer Integration" in readme
    assert "### Ephemeral Execution" in readme
    assert "EphemeralRunRequest" in readme
    assert "EphemeralRuntime" in readme
    assert "run_ephemeral" in readme
    assert "start with the one-shot path first" not in readme
    assert "### One-shot Execution" not in readme
