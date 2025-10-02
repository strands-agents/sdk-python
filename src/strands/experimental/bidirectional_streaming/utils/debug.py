"""Debug utilities for Strands bidirectional streaming.

Provides consistent debug logging across all bidirectional streaming components
with configurable output control matching the Nova Sonic tool use example.
"""

import datetime
import inspect
import time

# Debug logging system matching successful tool use example
DEBUG = False  # Disable debug logging for clean output like tool use example


def debug_print(message):
    """Print debug message with timestamp and function name."""
    if DEBUG:
        function_name = inspect.stack()[1].function
        if function_name == "time_it_async":
            function_name = inspect.stack()[2].function
        timestamp = "{:%Y-%m-%d %H:%M:%S.%f}".format(datetime.datetime.now())[:-3]
        print(f"{timestamp} {function_name} {message}")


def log_event(event_type, **context):
    """Log important events with structured context."""
    if DEBUG:
        function_name = inspect.stack()[1].function
        timestamp = "{:%Y-%m-%d %H:%M:%S.%f}".format(datetime.datetime.now())[:-3]
        context_str = " ".join([f"{k}={v}" for k, v in context.items()]) if context else ""
        print(f"{timestamp} {function_name} EVENT: {event_type} {context_str}")


def log_flow(step, details=""):
    """Log important flow steps without excessive detail."""
    if DEBUG:
        function_name = inspect.stack()[1].function
        timestamp = "{:%Y-%m-%d %H:%M:%S.%f}".format(datetime.datetime.now())[:-3]
        print(f"{timestamp} {function_name} FLOW: {step} {details}")


async def time_it_async(label, method_to_run):
    """Time asynchronous method execution."""
    start_time = time.perf_counter()
    result = await method_to_run()
    end_time = time.perf_counter()
    debug_print(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result
