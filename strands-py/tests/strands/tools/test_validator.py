from strands.tools import _validator
from strands.types.content import Message


def test_validate_and_prepare_tools():
    message: Message = {
        "role": "assistant",
        "content": [
            {"text": "value"},
            {"toolUse": {"toolUseId": "t1", "name": "test_tool", "input": {"key": "value"}}},
            {"toolUse": {"toolUseId": "t2-invalid"}},
        ],
    }

    tool_uses = []
    tool_results = []
    invalid_tool_use_ids = []

    _validator.validate_and_prepare_tools(message, tool_uses, tool_results, invalid_tool_use_ids)

    tru_tool_uses, tru_tool_results, tru_invalid_tool_use_ids = tool_uses, tool_results, invalid_tool_use_ids
    exp_tool_uses = [
        {
            "input": {
                "key": "value",
            },
            "name": "test_tool",
            "toolUseId": "t1",
        },
        {
            # This now happens in stream_messages
            # "name": "INVALID_TOOL_NAME",
            "toolUseId": "t2-invalid",
        },
    ]
    exp_tool_results = [
        {
            "content": [
                {
                    "text": "Error: tool name missing",
                },
            ],
            "status": "error",
            "toolUseId": "t2-invalid",
        },
    ]
    exp_invalid_tool_use_ids = ["t2-invalid"]

    assert tru_tool_uses == exp_tool_uses
    assert tru_tool_results == exp_tool_results
    assert tru_invalid_tool_use_ids == exp_invalid_tool_use_ids


def test_validate_and_prepare_tools_turns_malformed_input_into_tool_result():
    tool_use = {
        "toolUseId": "t1",
        "name": "search",
        "input": {},
        "inputParseError": "Invalid JSON in tool input for 'search'",
    }
    message: Message = {
        "role": "assistant",
        "content": [{"toolUse": tool_use}],
    }

    tool_uses = []
    tool_results = []
    invalid_tool_use_ids = []

    _validator.validate_and_prepare_tools(message, tool_uses, tool_results, invalid_tool_use_ids)

    assert invalid_tool_use_ids == ["t1"]
    assert tool_results == [
        {
            "toolUseId": "t1",
            "status": "error",
            "content": [{"text": "Error: Invalid JSON in tool input for 'search'"}],
        }
    ]
    # The marker is popped at consumption so it never persists in history.
    assert "inputParseError" not in tool_use


def test_validate_and_prepare_tools_uses_out_of_band_parse_errors():
    # The marker has already been stripped from the tool use; the error is carried out-of-band.
    tool_use = {"toolUseId": "t1", "name": "search", "input": {}}
    message: Message = {
        "role": "assistant",
        "content": [{"toolUse": tool_use}],
    }

    tool_uses = []
    tool_results = []
    invalid_tool_use_ids = []

    _validator.validate_and_prepare_tools(
        message,
        tool_uses,
        tool_results,
        invalid_tool_use_ids,
        input_parse_errors={"t1": "Invalid JSON in tool input for 'search'"},
    )

    assert invalid_tool_use_ids == ["t1"]
    assert tool_results == [
        {
            "toolUseId": "t1",
            "status": "error",
            "content": [{"text": "Error: Invalid JSON in tool input for 'search'"}],
        }
    ]


def test_strip_input_parse_errors():
    malformed_tool_use = {
        "toolUseId": "t1",
        "name": "search",
        "input": {},
        "inputParseError": "Invalid JSON in tool input for 'search'",
    }
    valid_tool_use = {"toolUseId": "t2", "name": "calc", "input": {"x": 1}}
    message: Message = {
        "role": "assistant",
        "content": [{"toolUse": malformed_tool_use}, {"toolUse": valid_tool_use}, {"text": "ignored"}],
    }

    parse_errors = _validator.strip_input_parse_errors(message)

    assert parse_errors == {"t1": "Invalid JSON in tool input for 'search'"}
    # The marker is removed from the message so it never persists in history.
    assert "inputParseError" not in malformed_tool_use
    assert valid_tool_use == {"toolUseId": "t2", "name": "calc", "input": {"x": 1}}
