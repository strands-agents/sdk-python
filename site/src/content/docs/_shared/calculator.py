"""Minimal calculator tool used by documentation examples.

Included via the mkdocs-snippets syntax:

    --8<-- "_shared/calculator.py:calculator"

Deliberately tiny: the examples that use it are demonstrating something else
(a model provider, streaming, evals), so the tool should stay out of the way.
It takes two operands rather than an expression string, which means there is
no parser and no path that evaluates model-supplied input as code.
"""

# --8<-- [start:calculator]
import operator

from strands import tool

_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
}


@tool
def calculator(a: float, b: float, op: str) -> float:
    """Apply an arithmetic operator to two numbers.

    Args:
        a: Left operand.
        b: Right operand.
        op: One of "+", "-", "*", "/", "**".
    """
    return _OPS[op](a, b)
# --8<-- [end:calculator]
