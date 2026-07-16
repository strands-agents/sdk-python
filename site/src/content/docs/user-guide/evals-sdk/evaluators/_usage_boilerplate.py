# Shared boilerplate for the "Basic Usage" examples on the evaluator pages.
# Included into the per-page code blocks via the mkdocs-style snippets plugin
# (see src/plugins/remark-mkdocs-snippets.ts). The evaluator-specific lines
# (imports, cases, Experiment construction) stay on each page.

# --8<-- [start:trace_task_function]
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()

def task_function(case: Case) -> dict:
    agent = Agent(
        trace_attributes={"session.id": case.session_id},
        callback_handler=None
    )
    response = agent(case.input)
    spans = telemetry.in_memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(spans, session_id=case.session_id)
    return {"output": str(response), "trajectory": session}
# --8<-- [end:trace_task_function]

# --8<-- [start:run_experiment]
async def main():
    report = await experiment.run_evaluations_async(task_function)
    report.run_display()

asyncio.run(main())
# --8<-- [end:run_experiment]

# --8<-- [start:user_task_function]
# Setup telemetry
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

# Define task function
def user_task_function(case: Case) -> dict:
    agent = Agent(
        trace_attributes={
            "gen_ai.conversation.id": case.session_id,
            "session.id": case.session_id
        },
        callback_handler=None
    )
    agent_response = agent(case.input)

    # Map spans to session
    finished_spans = memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id=case.session_id)

    return {"output": str(agent_response), "trajectory": session}
# --8<-- [end:user_task_function]
