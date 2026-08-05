
import ast
import asyncio
import operator

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
           ast.Div: operator.truediv, ast.USub: operator.neg}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            base, exp = ev(n.left), ev(n.right)
            if abs(exp) > 64:
                raise ValueError(f"exponent too large in {expression!r}: {exp}")
            return base**exp
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError(
            f"{expression!r} is not arithmetic; supported: + - * / ** and parentheses over numbers"
        )

    return str(ev(ast.parse(expression, mode="eval").body))

# Setup telemetry
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

# 1. Define a task function
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

# 2. Create test cases
test_cases = [
    Case[str, str](name="math-1", input="Calculate the square root of 144", metadata={"category": "math"}),
    Case[str, str](
        name="math-2",
        input="What is 25 * 4? can you use that output and then divide it by 4, then the final output should be squared. Give me the final value.",
        metadata={"category": "math"},
    ),
]

# 3. Create evaluators
evaluators = [ToolSelectionAccuracyEvaluator()]

# 4. Create an experiment
experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)

# 5. Run evaluations
async def main():
    report = await experiment.run_evaluations_async(user_task_function)
    report.run_display()


if __name__ == "__main__":
    asyncio.run(main())
