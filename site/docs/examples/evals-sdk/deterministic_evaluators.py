"""Deterministic Evaluators Example.

Fast, code-based evaluation without LLM judges.
"""


import operator

from strands import tool

_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
}


@tool
def calculator(a: float, b: float, op: str) -> float:
    """Apply an arithmetic operator to two numbers.

    Args:
        a: Left operand.
        b: Right operand.
        op: One of "+", "-", "*", "/", "**".
    """
    return _OPS[op](a, b)

def get_response_with_tools(case: Case) -> dict:
    agent = Agent(tools=[calculator], callback_handler=None)
    response = agent(case.input)
    trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(agent.messages)
    return {"output": str(response), "trajectory": trajectory}


tool_cases = [
    Case(name="calc", input="What is 15 * 23?", expected_trajectory=["calculator"]),
]

tool_experiment = Experiment(
    cases=tool_cases,
    evaluators=[ToolCalled(tool_name="calculator")],
)


async def main():
    report = await experiment.run_evaluations_async(get_response)
    report.run_display()
    tool_report = await tool_experiment.run_evaluations_async(get_response_with_tools)
    tool_report.run_display()


if __name__ == "__main__":
    asyncio.run(main())
