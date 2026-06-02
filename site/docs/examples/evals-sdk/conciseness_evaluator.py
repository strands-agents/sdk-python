"""Conciseness Evaluator Example.

Evaluates whether agent responses are concise and efficient.
"""

from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import ConcisenessEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()


def task_function(case: Case) -> dict:
    telemetry.in_memory_exporter.clear()
    agent = Agent(
        trace_attributes={"session.id": case.session_id},
        callback_handler=None,
    )
    response = agent(case.input)
    spans = telemetry.in_memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(spans, session_id=case.session_id)
    return {"output": str(response), "trajectory": session}


cases = [
    Case(name="simple-math", input="What is 2 + 2?"),
    Case(name="definition", input="Define photosynthesis in one sentence."),
]

experiment = Experiment(cases=cases, evaluators=[ConcisenessEvaluator()])
reports = experiment.run_evaluations(task_function)
reports[0].run_display()
