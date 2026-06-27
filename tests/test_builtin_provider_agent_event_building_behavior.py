from __future__ import annotations

import json

import pytest

from agent_runtime._builtin_provider_agent_event_building import (
    build_claude_agent_event,
    build_codex_agent_event,
    build_opencode_agent_event,
)


@pytest.mark.parametrize("event_type", ["item.started", "item.completed"])
def test_codex_agent_event_building_builds_agent_message_from_item_lifecycle_events(
    event_type: str,
) -> None:
    line = (
        json.dumps(
            {
                "type": event_type,
                "item": {
                    "type": "agent_message",
                    "content": "codex output",
                },
            }
        )
        + "\n"
    )

    event = build_codex_agent_event(line)

    assert event.type == "agent_message"
    assert event.display_message == "codex output"
    assert event.raw_provider_output == line


@pytest.mark.parametrize("event_type", ["item.started", "item.completed"])
def test_codex_agent_event_building_builds_tool_call_from_item_lifecycle_events(
    event_type: str,
) -> None:
    line = (
        json.dumps(
            {
                "type": event_type,
                "item": {
                    "type": "shell",
                    "name": "shell",
                    "arguments": {"command": "pwd"},
                },
            }
        )
        + "\n"
    )

    event = build_codex_agent_event(line)

    assert event.type == "agent_tool_call"
    assert event.display_message == 'shell({"command":"pwd"})'
    assert event.raw_provider_output == line


@pytest.mark.parametrize(
    ("item", "expected_message"),
    [
        (
            {
                "type": "shell",
                "name": "shell",
                "arguments": {"command": "pwd"},
                "input": {"command": "ignored"},
                "payload": {"command": "ignored"},
            },
            'shell({"command":"pwd"})',
        ),
        (
            {
                "type": "shell",
                "name": "shell",
                "input": {"command": "pwd"},
                "payload": {"command": "ignored"},
            },
            'shell({"command":"pwd"})',
        ),
        (
            {
                "type": "shell",
                "name": "shell",
                "payload": {"command": "pwd"},
            },
            'shell({"command":"pwd"})',
        ),
        (
            {
                "type": "shell",
                "name": "shell",
                "metadata": {"command": "pwd"},
            },
            'shell({"type":"shell","name":"shell","metadata":{"command":"pwd"}})',
        ),
    ],
)
def test_codex_agent_event_building_prefers_expected_tool_payload_fields(
    item: dict[str, object],
    expected_message: str,
) -> None:
    line = json.dumps({"type": "item.completed", "item": item}) + "\n"

    event = build_codex_agent_event(line)

    assert event.type == "agent_tool_call"
    assert event.display_message == expected_message
    assert event.raw_provider_output == line


def test_codex_agent_event_building_builds_turn_summary_from_turn_completed() -> None:
    line = (
        json.dumps(
            {
                "type": "turn.completed",
                "turn_id": "turn_123",
                "usage": {
                    "input_tokens": 120,
                    "cached_tokens": 30,
                    "output_tokens": 45,
                },
            }
        )
        + "\n"
    )

    event = build_codex_agent_event(line)

    assert event.type == "turn_summary"
    assert "120" in event.display_message
    assert "30" in event.display_message
    assert "45" in event.display_message
    assert event.raw_provider_output == line


def test_codex_agent_event_building_omits_missing_turn_summary_fields() -> None:
    line = (
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 0,
                    "cached_tokens": "30",
                    "output_tokens": None,
                },
            }
        )
        + "\n"
    )

    event = build_codex_agent_event(line)

    assert event.type == "turn_summary"
    assert event.display_message == "input_tokens=0"
    assert event.raw_provider_output == line


def test_codex_agent_event_building_builds_other_event_from_plain_text_line() -> None:
    line = "  permission denied: missing approval  \n"

    event = build_codex_agent_event(line)

    assert event.type == "other"
    assert event.display_message == "permission denied: missing approval"
    assert event.raw_provider_output == line


@pytest.mark.parametrize(
    ("line", "expected_message"),
    [
        (
            json.dumps({"type": "thread.started", "thread_id": "thread-123"}) + "\n",
            "thread.started",
        ),
        (
            json.dumps({"type": "item.started", "item": "not-an-object"}) + "\n",
            "item.started",
        ),
        (json.dumps({"detail": "missing type"}) + "\n", "other"),
        (json.dumps(["not", "an", "object"]) + "\n", "non_object"),
    ],
)
def test_codex_agent_event_building_preserves_other_descriptors_and_fallbacks(
    line: str,
    expected_message: str,
) -> None:
    event = build_codex_agent_event(line)

    assert event.type == "other"
    assert event.display_message == expected_message
    assert event.raw_provider_output == line


def test_claude_agent_event_building_builds_other_event_from_plain_text_line() -> None:
    line = "  Reading prompt from stdin...  \n"

    event = build_claude_agent_event(line)

    assert event.type == "other"
    assert event.display_message == "Reading prompt from stdin..."
    assert event.raw_provider_output == line


def test_claude_agent_event_building_builds_turn_summary_from_result() -> None:
    line = (
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 2345,
                "duration_api_ms": 2100,
                "is_error": False,
                "num_turns": 1,
                "result": "final output",
                "session_id": "sess_123",
                "total_cost_usd": 0.0123,
                "usage": {
                    "input_tokens": 120,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 5,
                    "output_tokens": 42,
                    "server_tool_use": {"web_search_requests": 1},
                },
            }
        )
        + "\n"
    )

    event = build_claude_agent_event(line)

    assert event.type == "turn_summary"
    assert "success" in event.display_message
    assert "2345" in event.display_message
    assert "0.0123" in event.display_message
    assert event.raw_provider_output == line


def test_claude_agent_event_building_omits_missing_turn_summary_fields() -> None:
    line = (
        json.dumps(
            {
                "type": "result",
                "subtype": "",
                "duration_ms": 0,
                "total_cost_usd": "0.0123",
            }
        )
        + "\n"
    )

    event = build_claude_agent_event(line)

    assert event.type == "turn_summary"
    assert event.display_message == "duration_ms=0"
    assert event.raw_provider_output == line


def test_claude_agent_event_building_preserves_tool_payload_shape_for_tool_only_lines() -> (
    None
):
    line = (
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {"path": "a.md"}},
                        {
                            "type": "tool_use",
                            "name": "Write",
                            "input": {"path": "b.md"},
                        },
                    ]
                },
            }
        )
        + "\n"
    )

    event = build_claude_agent_event(line)

    assert event.type == "agent_tool_call"
    assert event.display_message == (
        'Read([{"type":"tool_use","name":"Read","input":{"path":"a.md"}},'
        '{"type":"tool_use","name":"Write","input":{"path":"b.md"}}])'
    )
    assert event.raw_provider_output == line


def test_claude_agent_event_building_includes_subtype_and_cwd_in_system_init_display_message() -> (
    None
):
    line = (
        json.dumps(
            {
                "type": "system",
                "subtype": "system.init",
                "cwd": "/workspace/project",
            }
        )
        + "\n"
    )

    event = build_claude_agent_event(line)

    assert event.type == "other"
    assert event.display_message == "system.init cwd=/workspace/project"
    assert event.raw_provider_output == line


def test_claude_agent_event_building_includes_subtype_and_token_count_in_system_thinking_tokens_display_message() -> (
    None
):
    line = (
        json.dumps(
            {
                "type": "system",
                "subtype": "system.thinking_tokens",
                "estimated_tokens": 321,
            }
        )
        + "\n"
    )

    event = build_claude_agent_event(line)

    assert event.type == "other"
    assert event.display_message == "system.thinking_tokens tokens=321"
    assert event.raw_provider_output == line


def test_claude_agent_event_building_uses_subtype_name_for_unrecognized_system_subtype() -> (
    None
):
    line = (
        json.dumps(
            {
                "type": "system",
                "subtype": "system.custom_event",
                "cwd": "/workspace/project",
                "estimated_tokens": 321,
            }
        )
        + "\n"
    )

    event = build_claude_agent_event(line)

    assert event.type == "other"
    assert event.display_message == "system.custom_event"
    assert event.raw_provider_output == line


@pytest.mark.parametrize(
    ("line_payload", "expected_display_message"),
    [
        (
            {
                "type": "system",
                "subtype": "system.init",
            },
            "system.init",
        ),
        (
            {
                "type": "system",
                "subtype": "system.init",
                "cwd": "",
            },
            "system.init",
        ),
        (
            {
                "type": "system",
                "subtype": "system.thinking_tokens",
            },
            "system.thinking_tokens",
        ),
        (
            {
                "type": "system",
                "subtype": "system.thinking_tokens",
                "estimated_tokens": "321",
            },
            "system.thinking_tokens",
        ),
        (
            {
                "type": "system",
                "subtype": "system.thinking_tokens",
                "estimated_tokens": True,
            },
            "system.thinking_tokens",
        ),
    ],
)
def test_claude_agent_event_building_falls_back_to_subtype_name_without_valid_specialized_fields(
    line_payload: dict[str, object], expected_display_message: str
) -> None:
    line = json.dumps(line_payload) + "\n"

    event = build_claude_agent_event(line)

    assert event.type == "other"
    assert event.display_message == expected_display_message
    assert event.raw_provider_output == line


def test_opencode_agent_event_building_builds_expected_live_agent_events() -> None:
    cases = [
        (
            json.dumps(
                {
                    "type": "text",
                    "part": {
                        "type": "text",
                        "text": "hello from opencode",
                        "time": {"end": True},
                    },
                }
            )
            + "\n",
            "agent_message",
            "hello from opencode",
        ),
        (
            json.dumps(
                {
                    "type": "text",
                    "part": {
                        "type": "tool",
                        "name": "Read",
                        "input": {"path": "README.md"},
                    },
                }
            )
            + "\n",
            "agent_tool_call",
            'Read({"path":"README.md"})',
        ),
        (
            json.dumps(
                {
                    "type": "step_finish",
                    "step": {
                        "tokens": {
                            "input": 120,
                            "output": 45,
                            "reasoning": 12,
                            "cache": {"read": 30, "write": 8},
                        },
                        "cost": 0.0123,
                    },
                }
            )
            + "\n",
            "turn_summary",
            "input=120 | output=45 | reasoning=12 | cache_read=30 | cache_write=8 | cost_usd=0.0123",
        ),
        (
            json.dumps({"type": "session.status", "status": {"type": "idle"}}) + "\n",
            "other",
            "idle",
        ),
        (
            json.dumps({"type": "error", "error": {"name": "InternalServerError"}})
            + "\n",
            "other",
            "error",
        ),
        ("  not json  \n", "other", "not json"),
        (json.dumps({"type": "custom.event"}) + "\n", "other", "custom.event"),
    ]

    for line, expected_type, expected_message in cases:
        event = build_opencode_agent_event(line)
        assert event.type == expected_type
        assert event.display_message == expected_message
        assert event.raw_provider_output == line


def test_opencode_agent_event_building_omits_missing_turn_summary_fields() -> None:
    line = (
        json.dumps(
            {
                "type": "step_finish",
                "step": {
                    "tokens": {
                        "input": 0,
                        "output": "45",
                        "cache": {"read": 0, "write": None},
                    },
                    "cost": False,
                },
            }
        )
        + "\n"
    )

    event = build_opencode_agent_event(line)

    assert event.type == "turn_summary"
    assert event.display_message == "input=0 | cache_read=0"
    assert event.raw_provider_output == line


def test_opencode_agent_event_building_uses_text_payload_for_tool_part_without_input() -> (
    None
):
    line = (
        json.dumps(
            {
                "type": "text",
                "part": {
                    "type": "tool",
                    "name": "Read",
                    "text": '{"path":"README.md"}',
                },
            }
        )
        + "\n"
    )

    event = build_opencode_agent_event(line)

    assert event.type == "agent_tool_call"
    assert event.display_message == 'Read({"path":"README.md"})'
    assert event.raw_provider_output == line


def test_opencode_agent_event_building_keeps_session_status_descriptor_when_status_type_missing() -> (
    None
):
    line = json.dumps({"type": "session.status", "status": {}}) + "\n"

    event = build_opencode_agent_event(line)

    assert event.type == "other"
    assert event.display_message == "session.status"
    assert event.raw_provider_output == line


def test_opencode_agent_event_building_falls_back_to_text_descriptor_for_incomplete_text_part() -> (
    None
):
    line = (
        json.dumps(
            {
                "type": "text",
                "part": {
                    "type": "text",
                    "text": "still streaming",
                    "time": {"start": 1},
                },
            }
        )
        + "\n"
    )

    event = build_opencode_agent_event(line)

    assert event.type == "other"
    assert event.display_message == "text"
    assert event.raw_provider_output == line


@pytest.mark.parametrize(
    ("builder", "line"),
    [
        (build_claude_agent_event, '"text"\n'),
        (build_codex_agent_event, '"text"\n'),
        (build_opencode_agent_event, '"text"\n'),
    ],
)
def test_agent_event_building_keeps_non_object_descriptor(builder, line: str) -> None:
    event = builder(line)

    assert event.type == "other"
    assert event.display_message == "non_object"
    assert event.raw_provider_output == line
