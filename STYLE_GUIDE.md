# Style Guide

## Overview

The Strands Agents style guide aims to establish consistent formatting, naming conventions, and structure across all code in the repository. We strive to make our code clean, readable, and maintainable.

Where possible, we will codify these style guidelines into our linting rules and pre-commit hooks to automate enforcement and reduce the manual review burden.

## Log Formatting

The format for Strands Agents logs is as follows:

```python
logger.debug("field1=<%s>, field2=<%s>, ... | human readable message", field1, field2, ...)
```

### Guidelines

1. **Context**:
   - Add context as `<FIELD>=<VALUE>` pairs at the beginning of the log
     - Many log services (CloudWatch, Splunk, etc.) look for these patterns to extract fields for searching
   - Use `,`'s to separate pairs
   - Enclose values in `<>` for readability
     - This is particularly helpful in displaying empty values (`field=` vs `field=<>`)
   - Use `%s` for string interpolation as recommended by Python logging
     - This is an optimization to skip string interpolation when the log level is not enabled

1. **Messages**:
   - Add human-readable messages at the end of the log
   - Use lowercase for consistency
   - Avoid punctuation (periods, exclamation points, etc.) to reduce clutter
   - Keep messages concise and focused on a single statement
   - If multiple statements are needed, separate them with the pipe character (`|`)
     - Example: `"processing request | starting validation"`

### Examples

#### Good

```python
logger.debug("user_id=<%s>, action=<%s> | user performed action", user_id, action)
logger.info("request_id=<%s>, duration_ms=<%d> | request completed", request_id, duration)
logger.warning("attempt=<%d>, max_attempts=<%d> | retry limit approaching", attempt, max_attempts)
```

#### Poor

```python
# Avoid: No structured fields, direct variable interpolation in message
logger.debug(f"User {user_id} performed action {action}")

# Avoid: Inconsistent formatting, punctuation
logger.info("Request completed in %d ms.", duration)

# Avoid: No separation between fields and message
logger.warning("Retry limit approaching! attempt=%d max_attempts=%d", attempt, max_attempts)
```

By following these log formatting guidelines, we ensure that logs are both human-readable and machine-parseable, making debugging and monitoring more efficient.
