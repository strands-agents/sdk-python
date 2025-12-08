"""SteeringAction types for steering evaluation results.

Defines structured outcomes from steering handlers that determine how tool calls
should be handled. SteeringActions enable modular prompting by providing just-in-time
feedback rather than front-loading all instructions in monolithic prompts.

Flow:
    SteeringHandler.steer() → SteeringAction → BeforeToolCallEvent handling
                    ↓                     ↓                      ↓
              Evaluate context      Action type           Tool execution modified

SteeringAction types:
    Proceed: Tool executes immediately (no intervention needed)
    Guide: Tool cancelled, agent receives contextual feedback to explore alternatives
    Interrupt: Tool execution paused for human input via interrupt system

Extensibility:
    New action types can be added to the union. Always handle the default
    case in pattern matching to maintain backward compatibility.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Proceed(BaseModel):
    """Allow tool to execute immediately without intervention.

    The tool call proceeds as planned. The reason provides context
    for logging and debugging purposes.
    """

    type: Literal["proceed"] = "proceed"
    reason: str


class Guide(BaseModel):
    """Cancel tool and provide contextual feedback for agent to explore alternatives.

    The tool call is cancelled and the agent receives the reason as contextual
    feedback to help them consider alternative approaches while maintaining
    adaptive reasoning capabilities.
    """

    type: Literal["guide"] = "guide"
    reason: str


class Interrupt(BaseModel):
    """Pause tool execution for human input via interrupt system.

    The tool call is paused and human input is requested through Strands'
    interrupt system. The human can approve or deny the operation, and their
    decision determines whether the tool executes or is cancelled.
    """

    type: Literal["interrupt"] = "interrupt"
    reason: str


# SteeringAction union - extensible for future action types
# IMPORTANT: Always handle the default case when pattern matching
# to maintain backward compatibility as new action types are added
SteeringAction = Annotated[Proceed | Guide | Interrupt, Field(discriminator="type")]
