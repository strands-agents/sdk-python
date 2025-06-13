"""MetricsClient for OpenTelemetry integration.

This module provides meter capabilities using OpenTelemetry,
enabling metrics data to be sent to OTLP endpoints.
"""

import threading
from logging import getLogger

from opentelemetry.metrics import Counter, Meter

from ..telemetry import metrics_constants as constants
from ..telemetry.metrics import Meter as StrandsMeter

logger = getLogger(__name__)


class MetricsClient:
    """Creates a new instance of the MetricsClient class if it doesn't exist, otherwise returns the existing instance.

    :return: The instance of the MetricsClient class.
    """

    _instance = None
    _lock = threading.Lock()

    meter: Meter
    strands_agent_invocation_count: Counter

    def __init__(self) -> None:
        """Initialize a MetricsClient instance.

        Note: Initialization logic is intentionally placed in __new__ rather than __init__
        to ensure it only runs once when the singleton instance is created, not every
        time the class is instantiated.
        """
        pass

    def __new__(cls):
        """Create and initialize a new MetricsClient instance if none exists.

        Implements the singleton pattern by ensuring only one instance exists.
        The initialization logic (meter setup and instrument creation) is performed
        here rather than in __init__ to avoid reinitializing the singleton instance
        on subsequent instantiations.

        Returns:
        MetricsClient: The singleton instance of the MetricsClient class.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Creating Strands MetricsClient")
                    cls._instance = super(MetricsClient, cls).__new__(cls)
                    meter = StrandsMeter()
                    cls._instance.meter = meter.meter
                    cls._instance.create_instruments()
        return cls._instance

    def create_instruments(self):
        """Creates the OpenTelemetry Counter instruments."""
        if not self.meter:
            logger.warning("Meter is not initialized")
            return
        self.strands_agent_invocation_count = self.meter.create_counter(
            name=constants.STRANDS_AGENT_INVOCATION_COUNT, unit="Count"
        )
