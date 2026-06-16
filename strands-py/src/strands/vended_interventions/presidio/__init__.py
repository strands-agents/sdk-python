"""PII redaction intervention for Strands agents.

Detects and anonymizes personally identifiable information (PII) in tool I/O and
model context using `Microsoft Presidio <https://github.com/microsoft/presidio>`_,
wired to the ``Transform`` intervention action so redaction happens in-place before
content reaches the model, downstream tools, or logs.

Presidio is an optional dependency. Install it with::

    pip install 'strands-agents[presidio]'
    python -m spacy download en_core_web_lg

Example:
    ```python
    from strands import Agent
    from strands.vended_interventions.presidio import PresidioRedaction

    agent = Agent(
        interventions=[PresidioRedaction(entities=["EMAIL_ADDRESS", "PHONE_NUMBER"])],
    )

    # Tool I/O and model context now have detected PII redacted in-place.
    result = agent("Email the report to alice@example.com")
    ```
"""

from .presidio import PresidioRedaction

__all__ = ["PresidioRedaction"]
