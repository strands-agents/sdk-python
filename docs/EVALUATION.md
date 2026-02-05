# GenAI Evaluation with Strands OpenTelemetry

This document describes how to use the GenAI evaluation functionality in Strands, which follows the [OpenTelemetry GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/pull/2563) for capturing evaluation results.

## Overview

The evaluation system allows you to capture evaluation results as OpenTelemetry events attached to GenAI operation spans. This enables:

- **Standardized evaluation data**: Following OpenTelemetry GenAI Semantic Conventions
- **Tool interoperability**: Any OTEL-compliant tool can consume evaluation events
- **Trace correlation**: Evaluations are linked to the spans being evaluated
- **Flexible evaluation**: Support for custom evaluators and metrics

## Key Components

### EvaluationResult

Represents an evaluation result with the following attributes:

- `name`: Name of the evaluation metric (e.g., "relevance", "hallucination")
- `score`: Numeric score from the evaluator
- `score_label`: Human-readable interpretation of the score (optional)
- `reasoning`: Explanation from the evaluator (optional)
- `response_id`: Links eval to the completion when span linking isn't possible (optional)

### EvaluationTracer

Handles adding evaluation events to OpenTelemetry spans using the `gen_ai.evaluation.result` event name.

## Usage Examples

### Basic Usage

```python
from strands.telemetry import (
    add_relevance_evaluation,
    add_hallucination_evaluation,
    add_accuracy_evaluation,
    get_evaluation_tracer,
    EvaluationResult
)
from strands import Agent
from opentelemetry import trace

# Set up your agent
agent = Agent(model=your_model, tools=your_tools)

# Run agent and get current span
with trace.get_tracer(__name__).start_as_current_span("agent_evaluation") as span:
    result = await agent.run_async("What's the weather like?")
    response = str(result)
    
    # Add evaluation using convenience functions
    add_relevance_evaluation(
        span,
        score=0.9,
        label="highly_relevant",
        reasoning="Response directly addresses the weather query"
    )
    
    add_hallucination_evaluation(
        span,
        score=0.1,
        label="factual",
        reasoning="Response contains verifiable weather information"
    )
```

### Custom Evaluations

```python
from strands.telemetry import get_evaluation_tracer, EvaluationResult

# Create custom evaluation
evaluation_tracer = get_evaluation_tracer()

custom_evaluation = EvaluationResult(
    name="user_satisfaction",
    score=0.85,
    score_label="satisfied",
    reasoning="Response is clear and helpful"
)

evaluation_tracer.add_evaluation_event(span, custom_evaluation)
```

### Using Custom Evaluator Functions

```python
def sentiment_evaluator(response: str) -> dict:
    """Custom evaluator that returns sentiment score."""
    # Your evaluation logic here
    positive_words = ["good", "great", "excellent", "helpful"]
    score = sum(1 for word in positive_words if word in response.lower()) / len(positive_words)
    
    return {
        "score": score,
        "label": "positive" if score > 0.5 else "neutral",
        "reasoning": f"Found {int(score * len(positive_words))} positive indicators"
    }

# Use with evaluate_and_trace
evaluation_tracer.evaluate_and_trace(
    span,
    evaluator_func=sentiment_evaluator,
    content=response,
    evaluation_name="sentiment"
)
```

### Multiple Evaluations

```python
# Add multiple evaluations at once
evaluations = [
    EvaluationResult(name="completeness", score=0.8, label="complete"),
    EvaluationResult(name="helpfulness", score=0.9, label="helpful"),
    EvaluationResult(name="clarity", score=0.85, label="clear")
]

evaluation_tracer.add_multiple_evaluation_events(span, evaluations)
```

## Integration with Agent Spans

The evaluation system works seamlessly with Strands' existing tracing:

```python
from strands import Agent
from strands.telemetry import add_relevance_evaluation

# Agent automatically creates spans
agent = Agent(model=model, tools=tools, agent_name="evaluated_agent")
result = await agent.run_async("Your query here")

# Get the current span (agent span) and add evaluation
from opentelemetry import trace
current_span = trace.get_current_span()

if current_span and current_span.is_recording():
    add_relevance_evaluation(
        current_span,
        score=0.95,
        label="highly_relevant",
        reasoning="Perfect match for user query"
    )
```

## OpenTelemetry Event Format

Evaluation events follow the OpenTelemetry GenAI Semantic Conventions:

```json
{
  "event_name": "gen_ai.evaluation.result",
  "attributes": {
    "gen_ai.evaluation.name": "relevance",
    "gen_ai.evaluation.score": 0.85,
    "gen_ai.evaluation.score.label": "relevant",
    "gen_ai.evaluation.reasoning": "Query terms found in response",
    "gen_ai.response.id": "optional_response_id"
  }
}
```

## Evaluation Metrics

### Common Evaluation Types

The system provides convenience functions for common evaluation types:

- **Relevance**: How well the response addresses the query
- **Hallucination**: Whether the response contains factual inaccuracies
- **Accuracy**: Correctness of the information provided

### Custom Metrics

You can create evaluations for any metric:

```python
# Custom metrics examples
EvaluationResult(name="toxicity", score=0.05, label="safe")
EvaluationResult(name="coherence", score=0.9, label="coherent")
EvaluationResult(name="creativity", score=0.7, label="creative")
EvaluationResult(name="bias", score=0.2, label="minimal_bias")
```

## Best Practices

### 1. Use Descriptive Names
Choose clear, standardized names for your evaluation metrics:
- Good: "relevance", "hallucination", "accuracy"
- Avoid: "score1", "eval", "test"

### 2. Consistent Scoring
Use consistent scoring ranges across evaluations:
- 0.0 to 1.0 for percentages/probabilities
- Document your scoring system

### 3. Meaningful Labels
Provide human-readable labels that make sense to stakeholders:
- "highly_relevant", "somewhat_relevant", "not_relevant"
- "factual", "uncertain", "hallucinated"

### 4. Include Reasoning
Always provide reasoning when possible to make evaluations interpretable:
```python
EvaluationResult(
    name="relevance",
    score=0.8,
    label="relevant",
    reasoning="Response addresses 4 out of 5 key points from the query"
)
```

### 5. Attach to Appropriate Spans
Attach evaluations to the span being evaluated:
- Agent response evaluations → agent span
- Tool output evaluations → tool span
- Overall conversation evaluations → conversation span

## Observability Integration

### Viewing Evaluation Data

Evaluation events can be viewed in any OpenTelemetry-compatible observability tool:

- **Console**: Use `StrandsTelemetry().setup_console_exporter()`
- **OTLP**: Use `StrandsTelemetry().setup_otlp_exporter()` with your endpoint
- **Tools**: Jaeger, Zipkin, Datadog, New Relic, etc.

### Querying Evaluations

Since evaluations are standard OTEL events, you can query them like any other telemetry data:

```sql
-- Example query (syntax varies by tool)
SELECT 
  span_id,
  event_attributes['gen_ai.evaluation.name'] as metric,
  event_attributes['gen_ai.evaluation.score'] as score,
  event_attributes['gen_ai.evaluation.score.label'] as label
FROM traces 
WHERE event_name = 'gen_ai.evaluation.result'
  AND event_attributes['gen_ai.evaluation.name'] = 'relevance'
  AND CAST(event_attributes['gen_ai.evaluation.score'] AS FLOAT) < 0.5
```

## Environment Configuration

Set up OpenTelemetry environment variables to control where evaluation data is sent:

```bash
# Send to OTLP endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT="https://your-otlp-endpoint"
export OTEL_EXPORTER_OTLP_HEADERS="api-key=your-api-key"

# Service identification
export OTEL_SERVICE_NAME="my-agent-service"

# Enable GenAI conventions
export OTEL_SEMCONV_STABILITY_OPT_IN="gen_ai_latest_experimental"
```

## Error Handling

The evaluation system is designed to be non-intrusive:

- Failed evaluations are logged but don't crash the agent
- Non-recording spans are handled gracefully
- Invalid evaluator results are skipped with warnings

```python
# This won't crash even if the evaluator fails
evaluation_tracer.evaluate_and_trace(
    span,
    potentially_failing_evaluator,
    content,
    "risky_evaluation"
)
```

## Performance Considerations

- Evaluations are added as events, which are lightweight
- Evaluator functions run synchronously - use async patterns for expensive evaluations
- Consider sampling for high-volume scenarios
- Evaluation overhead is minimal compared to model inference

## Migration from Other Systems

If you're migrating from other evaluation systems:

1. **Map your metrics**: Convert existing metric names to standardized ones
2. **Normalize scores**: Ensure scores follow consistent ranges
3. **Add context**: Include reasoning and labels for better interpretability
4. **Update queries**: Adapt existing queries to use OTEL event format

This evaluation system provides a standardized, interoperable way to capture and analyze GenAI evaluation results within the broader OpenTelemetry ecosystem.