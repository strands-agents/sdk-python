
import ast
import operator

import asyncio

from strands import Agent, tool

from strands_evals import Case, Experiment
from strands_evals.evaluators import ToolSelectionAccuracyEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as "144 ** 0.5" or "450 / 120".

    Args:
        expression: The arithmetic expression to evaluate.
    """
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError(f"unsupported expression: {expression}")

    return str(ev(ast.parse(expression, mode="eval").body))

telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

async def async_example():
    """
    Demonstrates running evaluations asynchronously with run_evaluations_async.

    This example:
    1. Defines a task function that uses an agent with the calculator tool
    2. Creates test cases for math scenarios
    3. Creates a ToolSelectionAccuracyEvaluator
    4. Runs evaluations asynchronously and returns the report

    Returns:
        EvaluationReport: The evaluation results
    """

    ### Step 1: Define task ###
    def user_task_function(case: Case) -> dict:
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

    ### Step 2: Create test cases ###
    test_cases = [
        Case[str, str](name="math-1", input="Calculate the square root of 144", metadata={"category": "math"}),
        Case[str, str](
            name="math-2",
            input="What is 25 * 4? can you use that output and then divide it by 4, then the final output should be squared. Give me the final value.",
            metadata={"category": "math"},
        )
    ]

    evaluators = [ToolSelectionAccuracyEvaluator()]
    experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)
    report = await experiment.run_evaluations_async(user_task_function)
    return report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.evaluate_async
    report = asyncio.run(async_example())

    # report.to_file("tool_selection_accuracy_async")
    report.run_display(include_actual_trajectory=True)
