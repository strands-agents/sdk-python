#!/usr/bin/env python3
"""Simple example showing how to add evaluation to an existing Strands agent.

This example demonstrates the minimal changes needed to add evaluation
to your existing agent code.
"""

import asyncio
import logging
import os

# Standard Strands imports
from strands import Agent
from strands.models.anthropic import AnthropicModel
from strands.tools import tool

# Add evaluation imports
from strands.telemetry import (
    StrandsTelemetry,
    add_relevance_evaluation,
    add_accuracy_evaluation,
    get_evaluation_tracer,
    EvaluationResult
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        The result of the calculation
    """
    try:
        # Simple evaluation - in production, use a safer math parser
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


async def main():
    """Main function demonstrating evaluation integration."""
    
    # 1. Set up OpenTelemetry (add this to your existing setup)
    telemetry = StrandsTelemetry()
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        telemetry.setup_otlp_exporter()
    else:
        telemetry.setup_console_exporter()  # For local development
    
    # 2. Create your agent as usual
    model = AnthropicModel(model_id="claude-3-haiku-20240307")
    agent = Agent(
        model=model,
        tools=[calculate],
        agent_name="math_agent"
    )
    
    # 3. Run your agent and add evaluations
    queries = [
        "What is 15 + 27?",
        "Calculate the area of a circle with radius 5",
        "What's the weather like today?"  # This should get low relevance for a math agent
    ]
    
    evaluation_tracer = get_evaluation_tracer()
    
    for query in queries:
        logger.info(f"Processing: {query}")
        
        # Run the agent (this creates spans automatically)
        result = await agent.run_async(query)
        response = str(result)
        
        logger.info(f"Response: {response}")
        
        # Get the current span to add evaluations
        from opentelemetry import trace
        current_span = trace.get_current_span()
        
        if current_span and current_span.is_recording():
            # Add evaluations based on your criteria
            
            # 1. Relevance evaluation
            is_math_query = any(word in query.lower() for word in ['calculate', 'what is', '+', '-', '*', '/', 'area', 'radius'])
            has_math_response = any(word in response.lower() for word in ['result', 'calculate', 'area', 'equals'])
            
            if is_math_query and has_math_response:
                relevance_score = 0.9
                relevance_label = "highly_relevant"
                reasoning = "Math query answered with mathematical calculation"
            elif is_math_query:
                relevance_score = 0.5
                relevance_label = "partially_relevant"
                reasoning = "Math query but response may not contain calculation"
            else:
                relevance_score = 0.2
                relevance_label = "not_relevant"
                reasoning = "Non-math query for math agent"
            
            add_relevance_evaluation(
                current_span,
                score=relevance_score,
                label=relevance_label,
                reasoning=reasoning
            )
            
            # 2. Accuracy evaluation (simple heuristic)
            has_error = "error" in response.lower()
            has_result = "result" in response.lower() or any(char.isdigit() for char in response)
            
            if has_error:
                accuracy_score = 0.1
                accuracy_label = "error"
            elif has_result and is_math_query:
                accuracy_score = 0.9
                accuracy_label = "accurate"
            else:
                accuracy_score = 0.5
                accuracy_label = "uncertain"
            
            add_accuracy_evaluation(
                current_span,
                score=accuracy_score,
                label=accuracy_label,
                reasoning=f"Error detected: {has_error}, Result present: {has_result}"
            )
            
            # 3. Custom evaluation using EvaluationResult
            response_length = len(response.split())
            completeness_score = min(response_length / 20.0, 1.0)  # Normalize to 0-1
            
            completeness_evaluation = EvaluationResult(
                name="completeness",
                score=completeness_score,
                score_label="complete" if completeness_score > 0.7 else "incomplete",
                reasoning=f"Response length: {response_length} words"
            )
            
            evaluation_tracer.add_evaluation_event(current_span, completeness_evaluation)
            
            logger.info("Added evaluation events to trace")
    
    logger.info("Evaluation example completed!")
    
    # 4. Optional: Add summary evaluation across all queries
    from opentelemetry import trace
    with trace.get_tracer(__name__).start_as_current_span("session_summary") as summary_span:
        # Add session-level evaluations
        session_evaluation = EvaluationResult(
            name="session_quality",
            score=0.85,
            score_label="good_session",
            reasoning=f"Processed {len(queries)} queries successfully"
        )
        evaluation_tracer.add_evaluation_event(summary_span, session_evaluation)


if __name__ == "__main__":
    # Make sure you have ANTHROPIC_API_KEY set
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("Please set ANTHROPIC_API_KEY environment variable")
        logger.info("You can also modify this example to use other models like OpenAI, Ollama, etc.")
        exit(1)
    
    print("Running simple evaluation example...")
    print("This will show evaluation events in the console output.")
    print("Set OTEL_EXPORTER_OTLP_ENDPOINT to send to your observability platform.")
    
    asyncio.run(main())