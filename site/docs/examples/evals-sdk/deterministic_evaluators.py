"""Deterministic Evaluators Example.

Fast, code-based evaluation without LLM judges.
"""


import asyncio

from strands import Agent, tool
from strands_evals import Case, Experiment
from strands_evals.evaluators import Contains, Equals, StartsWith, ToolCalled

# --- Output evaluators ---


def get_response(case: Case) -> str:
    agent = Agent(callback_handler=None)
    return str(agent(case.input))


cases = [
    Case(
        name="capital",
        input="What is the capital of France? Reply with just the city name.",
        expected_output="Paris",
    ),
]

experiment = Experiment(
    cases=cases,
    evaluators=[
        Contains(value="Paris", case_sensitive=False),
        Equals(),  # compares against expected_output
    ],
)

# --- Trajectory evaluator ---

from strands_evals.extractors import tools_use_extractor

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression such as "sqrt(144)" or "450 / 120".

    Args:
        expression: The expression to evaluate.
    """
    import sympy

    return str(sympy.sympify(expression).evalf())

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
