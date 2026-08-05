
import ast
import operator

from strands import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as "144 ** 0.5" or "450 / 120".

    Args:
        expression: The arithmetic expression to evaluate.
    """

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exponent = ev(node.left), ev(node.right)
            if abs(exponent) > 64:  # an unbounded pow hangs uninterruptibly
                raise ValueError(f"exponent too large: {exponent}")
            return base**exponent
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"not arithmetic (+ - * / ** and parens only): {expression!r}")

    return str(ev(ast.parse(expression, mode="eval").body))

# Setup telemetry
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

# 1. Define a task function
def user_task_function(case: Case) -> dict:
    """Execute agent with tools and capture trajectory."""
    agent = Agent(
        # IMPORTANT: trace_attributes with session IDs are required when using StrandsInMemorySessionMapper
        # to prevent spans from different test cases from being mixed together in the memory exporter
        trace_attributes={"gen_ai.conversation.id": case.session_id, "session.id": case.session_id},
        tools=[calculator],
        callback_handler=None,
    )
    agent_response = agent(case.input)
    finished_spans = memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id=case.session_id)
    return {"output": str(agent_response), "trajectory": session}


# 2. Create test cases
test_cases = [
    Case[str, str](
        name="simple-calculation",
        input="Calculate the square root of 144",
        metadata={"category": "math", "difficulty": "easy"},
    ),
    Case[str, str](
        name="percentage-calculation",
        input="What's 20 percent of 250?",
        metadata={"category": "math", "difficulty": "easy"},
    ),
    Case[str, str](
        name="complex-calculation",
        input="I need to calculate 15 + 27, then multiply the result by 3, and finally subtract 10.",
        metadata={"category": "math", "difficulty": "medium"},
    ),
]

# 3. Create evaluators
# The evaluator will check if tool parameters are faithful to the context
evaluators = [ToolParameterAccuracyEvaluator()]

# 4. Create an experiment
experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)

# 5. Run evaluations
async def main():
    report = await experiment.run_evaluations_async(user_task_function)
    report.run_display()


if __name__ == "__main__":
    asyncio.run(main())
