import asyncio

from strands import Agent
from strands.vended_tools import http_request

from strands_evals import Case, Experiment
from strands_evals.evaluators import ToolSelectionAccuracyEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

async def async_example():
    """
    Demonstrates running evaluations asynchronously with run_evaluations_async.

    This example:
    1. Defines a task function that uses an agent with the http_request tool
    2. Creates test cases for weather scenarios
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
            tools=[http_request],
            callback_handler=None,
        )
        agent_response = agent(case.input)
        finished_spans = memory_exporter.get_finished_spans()
        mapper = StrandsInMemorySessionMapper()
        session = mapper.map_to_session(finished_spans, session_id=case.session_id)
        return {"output": str(agent_response), "trajectory": session}

    ### Step 2: Create test cases ###
    test_cases = [
        Case[str, str](
            name="weather-1",
            input="What is the current temperature in Seattle? Get it from "
            "https://api.open-meteo.com/v1/forecast?latitude=47.6&longitude=-122.3"
            "&current=temperature_2m",
            metadata={"category": "weather"},
        ),
        Case[str, str](
            name="weather-2",
            input="Is it warmer in Seattle or Miami right now? Check "
            "https://api.open-meteo.com/v1/forecast?latitude=47.6&longitude=-122.3"
            "&current=temperature_2m and "
            "https://api.open-meteo.com/v1/forecast?latitude=25.8&longitude=-80.2"
            "&current=temperature_2m",
            metadata={"category": "weather"},
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
