#!/usr/bin/env python3
"""Example demonstrating GenAI evaluation with Strands OpenTelemetry integration.

This example shows how to:
1. Set up OpenTelemetry tracing with evaluation support
2. Run an agent and capture evaluation results
3. Use both built-in and custom evaluators
4. View evaluation data in traces

The evaluation events follow the OpenTelemetry GenAI Semantic Conventions
and can be consumed by any OTEL-compliant observability tool.
"""

import asyncio
import logging
import os
from typing import Any

from strands import Agent
from strands.models.anthropic import AnthropicModel
from strands.telemetry import (
    StrandsTelemetry,
    add_accuracy_evaluation,
    add_hallucination_evaluation,
    add_relevance_evaluation,
    get_evaluation_tracer,
    get_tracer,
    EvaluationResult,
)
from strands.tools import tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example evaluator functions
def relevance_evaluator(response: str, query: str) -> dict[str, Any]:
    """Simple relevance evaluator based on keyword matching."""
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    
    # Calculate overlap
    overlap = len(query_words.intersection(response_words))
    total_query_words = len(query_words)
    
    if total_query_words == 0:
        score = 0.0
    else:
        score = overlap / total_query_words
    
    # Determine label
    if score >= 0.7:
        label = "highly_relevant"
    elif score >= 0.4:
        label = "somewhat_relevant"
    else:
        label = "not_relevant"
    
    return {
        "score": score,
        "label": label,
        "reasoning": f"Found {overlap}/{total_query_words} query words in response"
    }


def hallucination_evaluator(response: str, context: str = "") -> dict[str, Any]:
    """Simple hallucination evaluator based on factual claims."""
    # This is a simplified example - real evaluators would use more sophisticated methods
    suspicious_phrases = [
        "according to my knowledge",
        "i believe",
        "it seems like",
        "probably",
        "might be"
    ]
    
    response_lower = response.lower()
    suspicious_count = sum(1 for phrase in suspicious_phrases if phrase in response_lower)
    
    # Score based on suspicious phrases (0 = no hallucination, 1 = high hallucination risk)
    score = min(suspicious_count * 0.2, 1.0)
    
    if score <= 0.2:
        label = "factual"
    elif score <= 0.6:
        label = "uncertain"
    else:
        label = "likely_hallucinated"
    
    return {
        "score": score,
        "label": label,
        "reasoning": f"Found {suspicious_count} uncertain language patterns"
    }


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city or location to get weather for
        
    Returns:
        Weather information for the location
    """
    # Simulate weather API call
    return f"The weather in {location} is sunny with a temperature of 72°F"


async def run_agent_with_evaluation():
    """Run an agent and add evaluation events to the traces."""
    
    # Set up OpenTelemetry
    telemetry = StrandsTelemetry()
    
    # Enable console exporter to see traces
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        telemetry.setup_otlp_exporter()
    else:
        telemetry.setup_console_exporter()
    
    # Create agent
    model = AnthropicModel(model_id="claude-3-haiku-20240307")
    agent = Agent(
        model=model,
        tools=[get_weather],
        agent_name="weather_agent"
    )
    
    # Get tracers
    tracer = get_tracer()
    evaluation_tracer = get_evaluation_tracer()
    
    # Test queries
    queries = [
        "What's the weather like in San Francisco?",
        "Tell me about the climate in Tokyo",
        "Is it raining in London right now?"
    ]
    
    for query in queries:
        logger.info(f"Processing query: {query}")
        
        # Run the agent and capture the span
        with tracer.tracer.start_as_current_span(f"evaluate_query") as eval_span:
            # Set query context
            eval_span.set_attribute("query", query)
            
            # Run the agent
            result = await agent.run_async(query)
            response = str(result)
            
            logger.info(f"Agent response: {response}")
            
            # Add evaluation events to the span
            
            # 1. Using convenience functions
            relevance_result = relevance_evaluator(response, query)
            add_relevance_evaluation(
                eval_span,
                score=relevance_result["score"],
                label=relevance_result["label"],
                reasoning=relevance_result["reasoning"]
            )
            
            hallucination_result = hallucination_evaluator(response)
            add_hallucination_evaluation(
                eval_span,
                score=hallucination_result["score"],
                label=hallucination_result["label"],
                reasoning=hallucination_result["reasoning"]
            )
            
            # 2. Using the evaluation tracer directly
            accuracy_score = 0.9 if "weather" in response.lower() else 0.3
            accuracy_evaluation = EvaluationResult(
                name="accuracy",
                score=accuracy_score,
                score_label="accurate" if accuracy_score > 0.7 else "inaccurate",
                reasoning="Response contains expected weather information"
            )
            evaluation_tracer.add_evaluation_event(eval_span, accuracy_evaluation)
            
            # 3. Using the evaluate_and_trace helper
            evaluation_tracer.evaluate_and_trace(
                eval_span,
                evaluator_func=lambda text: len(text.split()) / 50.0,  # Simple length-based quality score
                content=response,
                evaluation_name="response_quality"
            )
            
            # 4. Adding multiple evaluations at once
            additional_evaluations = [
                EvaluationResult(
                    name="completeness",
                    score=0.8,
                    score_label="complete",
                    reasoning="Response addresses the main question"
                ),
                EvaluationResult(
                    name="helpfulness",
                    score=0.9,
                    score_label="helpful",
                    reasoning="Response provides actionable information"
                )
            ]
            evaluation_tracer.add_multiple_evaluation_events(eval_span, additional_evaluations)
            
            logger.info("Added evaluation events to trace")
    
    logger.info("Evaluation example completed!")


async def demonstrate_agent_span_evaluation():
    """Demonstrate adding evaluations to agent spans directly."""
    
    # Set up telemetry
    telemetry = StrandsTelemetry().setup_console_exporter()
    
    # Create agent with tracing enabled
    model = AnthropicModel(model_id="claude-3-haiku-20240307")
    agent = Agent(
        model=model,
        tools=[get_weather],
        agent_name="evaluated_agent"
    )
    
    # Run agent - this will create agent spans automatically
    result = await agent.run_async("What's the weather in New York?")
    
    # Get the current span (this would be the agent span)
    from opentelemetry import trace
    current_span = trace.get_current_span()
    
    if current_span and current_span.is_recording():
        # Add evaluation to the agent span
        add_relevance_evaluation(
            current_span,
            score=0.95,
            label="highly_relevant",
            reasoning="Response directly answers the weather query"
        )
        
        # Add custom evaluation using the tracer method
        tracer = get_tracer()
        tracer.add_evaluation_event(
            current_span,
            evaluation_name="user_satisfaction",
            score=0.9,
            score_label="satisfied",
            reasoning="Response is clear and informative"
        )
    
    logger.info(f"Agent result with evaluations: {result}")


if __name__ == "__main__":
    # Make sure you have ANTHROPIC_API_KEY set
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("Please set ANTHROPIC_API_KEY environment variable")
        exit(1)
    
    print("Running agent with evaluation example...")
    asyncio.run(run_agent_with_evaluation())
    
    print("\nRunning agent span evaluation example...")
    asyncio.run(demonstrate_agent_span_evaluation())