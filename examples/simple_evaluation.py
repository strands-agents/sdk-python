"""Simple example demonstrating basic GenAI evaluation support in Strands.

This example shows how to use the core evaluation functionality to add
evaluation events to OpenTelemetry spans following the GenAI semantic conventions.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from strands.telemetry.evaluation import EvaluationResult, get_evaluation_tracer

# Set up basic OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add console exporter to see the traces
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)


def main():
    """Demonstrate basic evaluation functionality."""
    print("=== Basic GenAI Evaluation Example ===\n")

    # Get the evaluation tracer
    evaluation_tracer = get_evaluation_tracer()

    # Create a span representing a GenAI operation
    with tracer.start_as_current_span("chat_completion") as span:
        # Simulate a model response
        model_response = "The capital of France is Paris."
        print(f"Model Response: {model_response}")

        # Create evaluation results
        relevance_result = EvaluationResult(
            name="relevance",
            score=0.95,
            score_label="highly_relevant",
            reasoning="Response directly answers the question about France's capital",
        )

        accuracy_result = EvaluationResult(
            name="accuracy",
            score=1.0,
            score_label="accurate",
            reasoning="Paris is indeed the capital of France",
        )

        # Add evaluation events to the span
        print("\nAdding evaluation events...")
        evaluation_tracer.add_evaluation_event(span, relevance_result)
        evaluation_tracer.add_evaluation_event(span, accuracy_result)

        print("✓ Added relevance evaluation (score: 0.95)")
        print("✓ Added accuracy evaluation (score: 1.0)")

    print("\n=== Example Complete ===")
    print("Check the console output above to see the OpenTelemetry spans with evaluation events.")


if __name__ == "__main__":
    main()