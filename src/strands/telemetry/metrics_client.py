import threading
from logging import getLogger

from opentelemetry.metrics import Counter, Meter

from ..telemetry import metrics_constants as constants
from ..telemetry.metrics import Meter as StrandsMeter

logger = getLogger(__name__)


class MetricsClient:
    """
    Creates a new instance of the ExtensionsMetricClient class if it doesn't exist, otherwise returns the existing instance.

    :return: The instance of the ExtensionsMetricClient class.
    """

    _instance = None
    _lock = threading.Lock()

    meter: Meter
    strands_agent_invocation_count: Counter

    def __init__(self) -> None:
        pass

    def __new__(cls):
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
        self.strands_agent_invocation_count = self.meter.create_counter(name=constants.STRANDS_AGENT_INVOCATION_COUNT, unit="Count")