"""Characterize the optional topic deployment assets."""

from pathlib import Path

import yaml

from tldw.llm import SYSTEM_PROMPT
from tldw.topics import load_topics
from tldw.validate_config import main

ROOT = Path(__file__).resolve().parents[1]


def test_example_matches_legacy_sources_and_prompt() -> None:
    """Keep the example compatible with the legacy configuration."""
    # Arrange
    catalog = yaml.safe_load((ROOT / "channels.yaml").read_text())
    ids = [entry["channel_id"] for entry in catalog["channels"]]
    # Act
    registry = load_topics(
        ROOT / "topics.example.yaml", environ={"TLDW_DISCORD_WEBHOOK_URL": "fake"}
    )
    # Assert
    assert ids and set(registry.channel_ids) == set(ids)
    assert registry.topics["default"].system_prompt == SYSTEM_PROMPT
    assert registry.topics["default"]._builtin == "trading"
    assert registry.topics["default"].webhook_env == "TLDW_DISCORD_WEBHOOK_URL"
    assert not registry.topics["legal"].enabled
    assert not registry.topics["legal"].channels
    assert registry.topics["legal"].webhook_url is None
    assert registry.topics["legal"].system_prompt == (ROOT / "prompts/legal.md").read_text()


def test_compose_overlay_is_opt_in() -> None:
    """Mount the sample and prompt only in the explicit overlay."""
    # Arrange
    base = yaml.safe_load((ROOT / "compose.yml").read_text())["services"]["tldw"]
    # Act
    overlay = yaml.safe_load((ROOT / "compose.topics.yml").read_text())["services"]["tldw"]
    # Assert
    assert "TLDW_TOPICS_FILE" not in base["environment"]
    assert overlay["environment"]["TLDW_TOPICS_FILE"] == "/config/topics.yaml"
    assert set(overlay["volumes"]) == {
        "./topics.example.yaml:/config/topics.yaml:ro",
        "./prompts:/config/prompts:ro",
    }


def test_kubernetes_patch_is_opt_in() -> None:
    """Reference secrets and config files without changing the base deployment."""
    # Arrange
    base = yaml.safe_load((ROOT / "k8s/deployment.yaml").read_text())
    # Act
    patch = yaml.safe_load((ROOT / "k8s/topics-patch.yaml").read_text())
    spec = patch["spec"]["template"]["spec"]
    container = spec["containers"][0]
    env = {item["name"]: item for item in container["env"]}
    # Assert
    assert base["spec"]["replicas"] == 1 and base["spec"]["strategy"]["type"] == "Recreate"
    assert "TLDW_TOPICS_FILE" not in str(base)
    assert patch["metadata"]["name"] == container["name"] == "tldw"
    assert env["TLDW_TOPICS_FILE"]["value"] == "/etc/tldw/topics.yaml"
    assert env["TLDW_DISCORD_WEBHOOK_URL"]["value"] is None
    assert env["TLDW_DISCORD_WEBHOOK_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "tldw-discord", "key": "default-webhook"
    }
    assert env["TLDW_LEGAL_DISCORD_WEBHOOK_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "tldw-discord", "key": "legal-webhook", "optional": True
    }
    assert spec["volumes"][0]["configMap"]["items"] == [
        {"key": "topics.yaml", "path": "topics.yaml"},
        {"key": "legal.md", "path": "prompts/legal.md"},
    ]
    assert container["volumeMounts"][0] == {
        "name": "topic-config", "mountPath": "/etc/tldw", "readOnly": True
    }


def test_container_assets_are_not_activated() -> None:
    """Copy optional assets to both stages without activating topics."""
    # Arrange
    text = (ROOT / "Containerfile").read_text()
    # Act
    copies = [line for line in text.splitlines() if line.startswith("COPY")]
    # Assert
    assert sum("topics.example.yaml" in line for line in copies) == 2
    assert sum("prompts" in line for line in copies) == 2
    assert not any(line.startswith("ENV ") and "TLDW_TOPICS_FILE" in line for line in text.splitlines())


def test_example_passes_offline_legacy_comparison(monkeypatch, capsys) -> None:
    """Preflight the example against the actual legacy source set."""
    # Arrange
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("TLDW_TOPICS_FILE", str(ROOT / "topics.example.yaml"))
    monkeypatch.setenv("TLDW_DISCORD_WEBHOOK_URL", "fake")
    monkeypatch.delenv("TLDW_CHANNEL_IDS", raising=False)
    # Act
    main(["--compare-legacy"])
    # Assert
    assert "Valid configuration" in capsys.readouterr().out
