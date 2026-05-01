def apply_execution_limit_defaults(mock):
    """Set execution-limit attributes on a mock agent that bypasses Agent.__init__."""
    mock.max_turns = None
    mock.max_token_budget = None
    mock._invocation_turn_count = 0
    mock._invocation_token_count = 0
