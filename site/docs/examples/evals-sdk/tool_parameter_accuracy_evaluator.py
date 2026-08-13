import asyncio

from strands import Agent
from strands.vended_tools import http_request

from strands_evals import Case, Experiment
from strands_evals.evaluators import ToolParameterAccuracyEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

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
        tools=[http_request],
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
        name="seattle-weather",
        input="What is the current temperature in Seattle? Get it from "
        "https://api.open-meteo.com/v1/forecast?latitude=47.6&longitude=-122.3"
        "&current=temperature_2m",
        metadata={"category": "weather", "difficulty": "easy"},
    ),
    Case[str, str](
        name="tokyo-weather",
        input="What is the current temperature in Tokyo? Get it from "
        "https://api.open-meteo.com/v1/forecast?latitude=35.7&longitude=139.7"
        "&current=temperature_2m",
        metadata={"category": "weather", "difficulty": "easy"},
    ),
    Case[str, str](
        name="weather-comparison",
        input="Is it warmer in Seattle or Miami right now? Check "
        "https://api.open-meteo.com/v1/forecast?latitude=47.6&longitude=-122.3"
        "&current=temperature_2m and "
        "https://api.open-meteo.com/v1/forecast?latitude=25.8&longitude=-80.2"
        "&current=temperature_2m",
        metadata={"category": "multi_tool", "difficulty": "medium"},
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
