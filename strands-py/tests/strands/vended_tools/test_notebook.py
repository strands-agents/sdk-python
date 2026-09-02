"""Tests for the notebook tool."""

from types import SimpleNamespace

import pytest

from strands.agent.state import AgentState
from strands.types.tools import ToolContext
from strands.vended_tools.notebook import notebook
from strands.vended_tools.notebook.notebook import (
    _MAX_NOTEBOOK_NAME_LENGTH,
    _MAX_NOTEBOOK_SIZE_BYTES,
    _MAX_NOTEBOOKS,
)
from strands.vended_tools.notebook.types import DEFAULT_NOTEBOOK_DESCRIPTION


def _fresh_context(initial_notebooks: dict[str, str] | None = None) -> tuple[AgentState, ToolContext]:
    """Build a fresh AgentState and ToolContext for a test.

    Args:
        initial_notebooks: Optional pre-populated notebooks map.

    Returns:
        A tuple of (state, tool_context).
    """
    state = AgentState({"notebooks": initial_notebooks} if initial_notebooks is not None else {"notebooks": {}})
    agent = SimpleNamespace(state=state)
    ctx = ToolContext(
        tool_use={"name": "notebook", "toolUseId": "test-id", "input": {}},
        agent=agent,
        invocation_state={},
    )
    return state, ctx


async def _invoke(ctx: ToolContext, input_: dict, alist) -> dict:
    """Invoke the default notebook tool through the async stream() path."""
    events = await alist(notebook.stream({"toolUseId": "t", "input": input_}, {"agent": ctx.agent}))
    return events[-1]["tool_result"]


class TestCreate:
    """The ``create`` operation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name,expected_name",
        [
            (None, "default"),
            ("notes", "notes"),
        ],
    )
    async def test_creates_empty_notebook_stream_envelope(self, name, expected_name, alist):
        # Verifies the tool_result envelope shape through the stream() path.
        state, ctx = _fresh_context()
        input_ = {"mode": "create"}
        if name is not None:
            input_["name"] = name
        tru_result = await _invoke(ctx, input_, alist)
        exp_result = {
            "toolUseId": "t",
            "status": "success",
            "content": [{"text": f"Created notebook '{expected_name}' (empty)"}],
        }
        assert tru_result == exp_result
        assert state.get("notebooks")[expected_name] == ""

    @pytest.mark.asyncio
    async def test_creates_notebook_with_initial_content(self):
        state, ctx = _fresh_context()
        content = "# My Notes\n\nFirst entry"
        result = await notebook(mode="create", name="notes", new_str=content, tool_context=ctx)
        assert result == "Created notebook 'notes' with specified content"
        assert state.get("notebooks")["notes"] == content

    @pytest.mark.asyncio
    async def test_overwrites_existing_notebook(self):
        state, ctx = _fresh_context({"notes": "Old content"})
        result = await notebook(mode="create", name="notes", new_str="New content", tool_context=ctx)
        assert result == "Created notebook 'notes' with specified content"
        assert state.get("notebooks")["notes"] == "New content"


class TestList:
    """The ``list`` operation."""

    @pytest.mark.asyncio
    async def test_lists_default_when_initialized(self):
        _, ctx = _fresh_context({"default": ""})
        tru_result = await notebook(mode="list", tool_context=ctx)
        assert tru_result == "Available notebooks:\n- default: Empty"

    @pytest.mark.asyncio
    async def test_lists_multiple_with_line_counts(self):
        _, ctx = _fresh_context(
            {
                "default": "",
                "notes": "Line 1\nLine 2\nLine 3",
                "todo": "Single line",
            }
        )
        result = await notebook(mode="list", tool_context=ctx)
        assert "default: Empty" in result
        assert "notes: 3 lines" in result
        assert "todo: 1 lines" in result


class TestRead:
    """The ``read`` operation."""

    @pytest.mark.asyncio
    async def test_reads_entire_notebook(self):
        _, ctx = _fresh_context({"notes": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"})
        assert await notebook(mode="read", name="notes", tool_context=ctx) == "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

    @pytest.mark.asyncio
    async def test_reads_empty_notebook(self):
        _, ctx = _fresh_context({"empty": ""})
        assert await notebook(mode="read", name="empty", tool_context=ctx) == "Notebook 'empty' is empty"

    @pytest.mark.asyncio
    async def test_reads_specific_line_range(self):
        _, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"})
        result = await notebook(mode="read", read_range=[2, 4], tool_context=ctx)
        assert result == "2: Line 2\n3: Line 3\n4: Line 4"

    @pytest.mark.parametrize(
        "read_range,expected",
        [
            ([-3, 5], "3: Line 3\n4: Line 4\n5: Line 5"),
            ([1, -2], "1: Line 1\n2: Line 2\n3: Line 3\n4: Line 4"),
            ([-2, -1], "4: Line 4\n5: Line 5"),
        ],
    )
    @pytest.mark.asyncio
    async def test_reads_range_with_negative_indices(self, read_range, expected):
        _, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"})
        assert await notebook(mode="read", read_range=read_range, tool_context=ctx) == expected

    @pytest.mark.asyncio
    async def test_reads_out_of_range(self):
        _, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"})
        result = await notebook(mode="read", read_range=[10, 20], tool_context=ctx)
        assert "No lines found in range [10, 20]" in result
        assert "5 line(s)" in result


class TestWriteReplace:
    """The ``write`` operation in string-replacement mode."""

    @pytest.mark.asyncio
    async def test_replaces_text_default_notebook(self):
        state, ctx = _fresh_context({"default": "# Todo List\n\n[ ] Task 1\n[ ] Task 2\n[x] Task 3"})
        result = await notebook(mode="write", old_str="[ ] Task 1", new_str="[x] Task 1", tool_context=ctx)
        assert result == "Replaced text in notebook 'default'"
        assert state.get("notebooks")["default"] == "# Todo List\n\n[x] Task 1\n[ ] Task 2\n[x] Task 3"

    @pytest.mark.asyncio
    async def test_replaces_text_custom_notebook(self):
        state, ctx = _fresh_context({"notes": "Original text"})
        result = await notebook(mode="write", name="notes", old_str="Original", new_str="Updated", tool_context=ctx)
        assert result == "Replaced text in notebook 'notes'"
        assert state.get("notebooks")["notes"] == "Updated text"

    @pytest.mark.asyncio
    async def test_replaces_multiline_text(self):
        state, ctx = _fresh_context({"default": "# Todo List\n\n[ ] Task 1\n[ ] Task 2\n[x] Task 3"})
        result = await notebook(
            mode="write",
            old_str="[ ] Task 1\n[ ] Task 2",
            new_str="[x] Task 1\n[x] Task 2",
            tool_context=ctx,
        )
        assert result == "Replaced text in notebook 'default'"
        assert state.get("notebooks")["default"] == "# Todo List\n\n[x] Task 1\n[x] Task 2\n[x] Task 3"

    @pytest.mark.asyncio
    async def test_preserves_dollar_sign_patterns_literally(self):
        state, ctx = _fresh_context({"default": "const value = getPrice()"})
        result = await notebook(mode="write", old_str="getPrice()", new_str="$& is not $1 or $$", tool_context=ctx)
        assert result == "Replaced text in notebook 'default'"
        assert state.get("notebooks")["default"] == "const value = $& is not $1 or $$"

    @pytest.mark.asyncio
    async def test_replaces_only_first_occurrence(self):
        state, ctx = _fresh_context({"default": "a b a b"})
        await notebook(mode="write", old_str="a", new_str="X", tool_context=ctx)
        assert state.get("notebooks")["default"] == "X b a b"

    @pytest.mark.asyncio
    async def test_missing_old_string_raises(self):
        _, ctx = _fresh_context({"default": "# Todo List\n\n[ ] Task 1"})
        with pytest.raises(ValueError, match="String 'Nonexistent' not found in notebook 'default'"):
            await notebook(mode="write", old_str="Nonexistent", new_str="New", tool_context=ctx)


class TestWriteInsert:
    """The ``write`` operation in line-insertion mode."""

    @pytest.mark.asyncio
    async def test_inserts_after_line_number(self):
        state, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3"})
        result = await notebook(mode="write", insert_line=2, new_str="Inserted line", tool_context=ctx)
        assert result == "Inserted text at line 3 in notebook 'default'"
        assert state.get("notebooks")["default"] == "Line 1\nLine 2\nInserted line\nLine 3"

    @pytest.mark.asyncio
    async def test_inserts_at_beginning_line_zero(self):
        state, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3"})
        result = await notebook(mode="write", insert_line=0, new_str="First line", tool_context=ctx)
        assert result == "Inserted text at line 1 in notebook 'default'"
        assert state.get("notebooks")["default"] == "First line\nLine 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    async def test_inserts_after_last_line_via_negative_index(self):
        state, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3"})
        result = await notebook(mode="write", insert_line=-1, new_str="Last line", tool_context=ctx)
        assert result == "Inserted text at line 4 in notebook 'default'"
        assert state.get("notebooks")["default"] == "Line 1\nLine 2\nLine 3\nLast line"

    @pytest.mark.asyncio
    async def test_inserts_after_negative_index(self):
        state, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3"})
        result = await notebook(mode="write", insert_line=-2, new_str="Before last", tool_context=ctx)
        assert result == "Inserted text at line 3 in notebook 'default'"
        assert state.get("notebooks")["default"] == "Line 1\nLine 2\nBefore last\nLine 3"

    @pytest.mark.asyncio
    async def test_inserts_after_text_search(self):
        state, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3"})
        result = await notebook(mode="write", insert_line="Line 1", new_str="After Line 1", tool_context=ctx)
        assert result == "Inserted text at line 2 in notebook 'default'"
        assert state.get("notebooks")["default"] == "Line 1\nAfter Line 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    async def test_inserts_after_partial_text_match(self):
        state, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3"})
        result = await notebook(mode="write", insert_line="2", new_str="After match", tool_context=ctx)
        assert result == "Inserted text at line 3 in notebook 'default'"
        assert state.get("notebooks")["default"] == "Line 1\nLine 2\nAfter match\nLine 3"

    @pytest.mark.asyncio
    async def test_bool_insert_line_raises(self):
        _, ctx = _fresh_context({"default": "Line 1\nLine 2"})
        with pytest.raises(ValueError, match="insert_line"):
            await notebook(mode="write", insert_line=True, new_str="x", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_search_not_found_raises(self):
        _, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3"})
        with pytest.raises(ValueError, match="Text 'Nonexistent' not found in notebook 'default'"):
            await notebook(mode="write", insert_line="Nonexistent", new_str="New line", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_line_number_out_of_range_raises(self):
        _, ctx = _fresh_context({"default": "Line 1\nLine 2\nLine 3"})
        with pytest.raises(ValueError, match="Line number out of range"):
            await notebook(mode="write", insert_line=100, new_str="New line", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_inserts_into_custom_notebook(self):
        state, ctx = _fresh_context({"notes": "First\nSecond"})
        result = await notebook(mode="write", name="notes", insert_line=1, new_str="Middle", tool_context=ctx)
        assert result == "Inserted text at line 2 in notebook 'notes'"
        assert state.get("notebooks")["notes"] == "First\nMiddle\nSecond"


class TestWriteAppend:
    """The ``write`` operation in append mode (new_str alone, no old_str or insert_line)."""

    @pytest.mark.asyncio
    async def test_appends_to_non_empty_notebook(self):
        state, ctx = _fresh_context({"default": "Line 1"})
        result = await notebook(mode="write", new_str="Line 2", tool_context=ctx)
        assert result == "Appended text to notebook 'default'"
        assert state.get("notebooks")["default"] == "Line 1\nLine 2"

    @pytest.mark.asyncio
    async def test_appends_to_empty_notebook(self):
        state, ctx = _fresh_context({"default": ""})
        result = await notebook(mode="write", new_str="First line", tool_context=ctx)
        assert result == "Appended text to notebook 'default'"
        assert state.get("notebooks")["default"] == "First line"

    @pytest.mark.asyncio
    async def test_appends_to_custom_notebook(self):
        state, ctx = _fresh_context({"notes": "existing"})
        result = await notebook(mode="write", name="notes", new_str="appended", tool_context=ctx)
        assert result == "Appended text to notebook 'notes'"
        assert state.get("notebooks")["notes"] == "existing\nappended"

    @pytest.mark.asyncio
    async def test_no_double_newline_when_content_ends_with_newline(self):
        # If the notebook already ends with \n, no extra separator is added.
        state, ctx = _fresh_context({"default": "Line 1\n"})
        await notebook(mode="write", new_str="Line 2", tool_context=ctx)
        assert state.get("notebooks")["default"] == "Line 1\nLine 2"

    @pytest.mark.asyncio
    async def test_empty_new_str_is_noop(self):
        state, ctx = _fresh_context({"default": "unchanged"})
        result = await notebook(mode="write", new_str="", tool_context=ctx)
        assert result == "No changes made to notebook 'default'"
        assert state.get("notebooks")["default"] == "unchanged"


class TestClear:
    """The ``clear`` operation."""

    @pytest.mark.asyncio
    async def test_clears_default_notebook(self):
        state, ctx = _fresh_context({"default": "Some content"})
        result = await notebook(mode="clear", tool_context=ctx)
        assert result == "Cleared notebook 'default'"
        assert state.get("notebooks")["default"] == ""

    @pytest.mark.asyncio
    async def test_clears_custom_notebook(self):
        state, ctx = _fresh_context({"notes": "More content"})
        result = await notebook(mode="clear", name="notes", tool_context=ctx)
        assert result == "Cleared notebook 'notes'"
        assert state.get("notebooks")["notes"] == ""

    @pytest.mark.asyncio
    async def test_clearing_does_not_affect_other_notebooks(self):
        state, ctx = _fresh_context({"default": "Some content", "notes": "More content"})
        await notebook(mode="clear", name="notes", tool_context=ctx)
        assert state.get("notebooks")["default"] == "Some content"


class TestStatePersistence:
    """Notebook state persists across operations on the same agent."""

    @pytest.mark.asyncio
    async def test_persists_across_operations(self):
        state, ctx = _fresh_context()
        await notebook(mode="create", name="notes", new_str="Initial", tool_context=ctx)
        assert state.get("notebooks")["notes"] == "Initial"

        await notebook(mode="write", name="notes", old_str="Initial", new_str="Initial\nAdded", tool_context=ctx)
        assert state.get("notebooks")["notes"] == "Initial\nAdded"

        content = await notebook(mode="read", name="notes", tool_context=ctx)
        assert content == "Initial\nAdded"
        assert state.get("notebooks")["notes"] == "Initial\nAdded"

    @pytest.mark.asyncio
    async def test_read_does_not_mutate_state(self):
        # If a sibling tool grew state past the cap, a pure read must not fail.
        oversized = {f"nb-{i}": "a" * (_MAX_NOTEBOOK_SIZE_BYTES // 2) for i in range(20)}
        state, ctx = _fresh_context(oversized)
        notebooks_before = state.get("notebooks").copy()
        version_before = state._get_version()
        await notebook(mode="read", name="nb-0", tool_context=ctx)
        assert state.get("notebooks") == notebooks_before
        assert state._get_version() == version_before

    @pytest.mark.asyncio
    async def test_list_does_not_mutate_state_when_empty(self):
        # Empty state materializes a default in-memory but must not persist it on `list`.
        state, ctx = _fresh_context()
        await notebook(mode="list", tool_context=ctx)
        assert state.get("notebooks") == {}

    @pytest.mark.asyncio
    async def test_rejects_malformed_state(self):
        state = AgentState({"notebooks": {"good": ["not", "a", "string"]}})
        agent = SimpleNamespace(state=state)
        ctx = ToolContext(
            tool_use={"name": "notebook", "toolUseId": "t", "input": {}},
            agent=agent,
            invocation_state={},
        )
        with pytest.raises(ValueError, match="Malformed notebooks state"):
            await notebook(mode="read", tool_context=ctx)


class TestValidationErrors:
    """Validation of inputs at the tool boundary."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mode,kwargs",
        [
            ("read", {"name": "missing"}),
            ("write", {"name": "missing", "old_str": "Old", "new_str": "New"}),
            ("write", {"name": "missing", "new_str": "x"}),
            ("clear", {"name": "missing"}),
        ],
    )
    async def test_missing_notebook_raises(self, mode, kwargs):
        _, ctx = _fresh_context()
        with pytest.raises(ValueError, match="Notebook 'missing' not found"):
            await notebook(mode=mode, tool_context=ctx, **kwargs)

    @pytest.mark.asyncio
    async def test_rejects_write_without_new_str_for_replacement(self):
        _, ctx = _fresh_context()
        with pytest.raises(ValueError):
            await notebook(mode="write", old_str="Old", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_rejects_write_without_new_str_for_insertion(self):
        _, ctx = _fresh_context()
        with pytest.raises(ValueError):
            await notebook(mode="write", insert_line=1, tool_context=ctx)

    @pytest.mark.asyncio
    async def test_rejects_write_without_operation_params(self):
        _, ctx = _fresh_context()
        with pytest.raises(ValueError):
            await notebook(mode="write", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_rejects_write_with_both_old_str_and_insert_line(self):
        # Ambiguous: both replacement and insertion anchors provided. Reject rather
        # than silently preferring one mode to avoid corrupting the notebook.
        _, ctx = _fresh_context({"default": "Line 1\nLine 2"})
        with pytest.raises(ValueError, match="ambiguous"):
            await notebook(
                mode="write",
                old_str="Line 1",
                new_str="Replaced",
                insert_line=0,
                tool_context=ctx,
            )

    @pytest.mark.asyncio
    async def test_rejects_read_range_wrong_length(self):
        _, ctx = _fresh_context({"default": "Line 1\nLine 2"})
        with pytest.raises(ValueError, match="two integers"):
            await notebook(mode="read", read_range=[1], tool_context=ctx)


class TestNameConfinement:
    """Notebook-name validation prevents path-like keys from leaking through state."""

    @pytest.mark.parametrize(
        "bad_name",
        [
            "../etc/passwd",
            "..\\evil",
            "/absolute/path",
            "nested/name",
            "back\\slash",
            "..",
            ".",
            "notes\x00nul",  # NUL-byte smuggling
            "   ",  # whitespace-only
            " leading",
            "trailing ",
        ],
    )
    @pytest.mark.asyncio
    async def test_rejects_path_like_names(self, bad_name):
        _, ctx = _fresh_context()
        with pytest.raises(ValueError):
            await notebook(mode="create", name=bad_name, tool_context=ctx)

    @pytest.mark.asyncio
    async def test_rejects_empty_name(self):
        _, ctx = _fresh_context()
        with pytest.raises(ValueError, match="non-empty"):
            await notebook(mode="create", name="", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_rejects_overly_long_name(self):
        _, ctx = _fresh_context()
        long_name = "a" * (_MAX_NOTEBOOK_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="maximum length"):
            await notebook(mode="create", name=long_name, tool_context=ctx)


class TestSessionCaps:
    """Session-scoped memory caps guard against runaway state growth."""

    @pytest.mark.asyncio
    async def test_rejects_too_many_notebooks(self):
        # Pre-fill state right up to the limit, then try to create one more.
        initial = {f"nb-{i}": "" for i in range(_MAX_NOTEBOOKS)}
        _, ctx = _fresh_context(initial)
        with pytest.raises(ValueError, match="notebook count"):
            await notebook(mode="create", name="over-the-line", tool_context=ctx)
        # Cap failure must not persist the rejected notebook.
        assert "over-the-line" not in (_.get("notebooks") or {})

    @pytest.mark.asyncio
    async def test_rejects_notebook_content_over_size_limit(self):
        _, ctx = _fresh_context()
        oversized = "a" * (_MAX_NOTEBOOK_SIZE_BYTES + 1)
        with pytest.raises(ValueError, match="maximum of"):
            await notebook(mode="create", name="big", new_str=oversized, tool_context=ctx)
        # Cap failure must not persist the oversized notebook.
        assert "big" not in (_.get("notebooks") or {})


class TestToolMetadata:
    """Tool names, descriptions, and input schemas."""

    def test_default_name(self):
        assert notebook.tool_name == "notebook"

    def test_default_description(self):
        assert notebook.tool_spec["description"] == DEFAULT_NOTEBOOK_DESCRIPTION

    def test_schema_excludes_context(self):
        props = notebook.tool_spec["inputSchema"]["json"]["properties"]
        assert "mode" in props
        assert "tool_context" not in props
