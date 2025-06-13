"""OpenTelemetry configuration and setup utilities for Strands agents.

This module provides centralized configuration and initialization functionality
for OpenTelemetry components and other telemetry infrastructure shared across Strands applications.
"""

from importlib.metadata import version

from opentelemetry.sdk.resources import Resource


def get_otel_resource() -> Resource:
    """Create a standard OpenTelemetry resource with service information.

    This function implements a singleton pattern - it will return the same
    Resource object for the same service_name parameter.

    Args:
        service_name: Name of the service for OpenTelemetry.

    Returns:
        Resource object with standard service information.
    """
    resource = Resource.create(
        {
            "service.name": __name__,
            "service.version": version("strands-agents"),
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
        }
    )

    return resource
