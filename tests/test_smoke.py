import importlib
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


def test_readme_stays_consumer_facing() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "[the public API reference](docs/public-api.md)" in readme
    assert "## Layout" not in readme
    assert "## Development" not in readme
    assert "pip install -e" not in readme
    assert "pytest" not in readme
    assert "docs/adr" not in readme


def test_public_api_reference_documents_runtime_surface_tiers() -> None:
    public_api = (
        Path(__file__).resolve().parents[1] / "docs" / "public-api.md"
    ).read_text(encoding="utf-8")

    assert "# agent_runtime Public API" in public_api
    assert "## Consumer Surface" in public_api
    assert "## Adapter Surface" in public_api
    assert "## Advanced Focused Seams" in public_api
    assert "not an inventory of every importable symbol" in public_api
    assert "compatibility aliases are intentionally absent" in public_api


def test_public_api_reference_mentions_documented_exports() -> None:
    public_api = (
        Path(__file__).resolve().parents[1] / "docs" / "public-api.md"
    ).read_text(encoding="utf-8")
    documented_modules = [
        "agent_runtime",
        "agent_runtime.runtime",
        "agent_runtime.contracts",
        "agent_runtime.execution_contracts",
        "agent_runtime.errors",
        "agent_runtime.provider_errors",
        "agent_runtime.provider_output",
        "agent_runtime.provider_session_adapter",
        "agent_runtime.session",
        "agent_runtime.session_planning",
    ]

    for module_name in documented_modules:
        module = importlib.import_module(module_name)
        for exported_name in module.__all__:
            assert exported_name in public_api

    for curated_name in [
        "ServiceRegistry",
        "AgentInvocationLog",
        "LogicalAgentInvocationLog",
        "WorkInvocationLog",
    ]:
        assert curated_name in public_api
